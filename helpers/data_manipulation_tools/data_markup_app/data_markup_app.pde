/* ============================================================
   SIMPLE LABELING TOOL FOR VOICE DATASETS
   Left side: audio player + navigation
   Right side: sliders for F/M/A
   Auto-load + auto-save on speaker switch
   ============================================================ */

import ddf.minim.*;
import java.util.Arrays;
import java.util.Comparator;

// ── Minim ────────────────────────────────────────────────────
Minim minim;
AudioPlayer player;

// ── Paths ────────────────────────────────────────────────────
final String DATASET_PATH = "../../../data/2_structure_and_markup/dataset/";
final String CSV_PATH     = "../../../data/2_structure_and_markup/dataset/speakers_info.csv";

// ── CSV columns ──────────────────────────────────────────────
final String COL_SPEAKER     = "speaker_name";
final String COL_FEMININITY  = "femininity";
final String COL_MASCULINITY = "masculinity";
final String COL_ATYPICALITY = "atypicality";

// ── Dataset ──────────────────────────────────────────────────
String[]   speakers;
String[][] speakerFiles;
int speakerIndex = 0;
int fileIndex    = 0;
Table infoTable;

// ── Slider values ────────────────────────────────────────────
float fem = 0, mas = 0, atyp = 0;
boolean changed = false;

// ── UI constants ─────────────────────────────────────────────
final int SLIDER_W = 380;
final int SLIDER_H = 68;
final int UI_LEFT  = 10;
final int UI_RIGHT = 310;

// ── Gesture state ────────────────────────────────────────────
String currentGestureTargetId = "";


/* ============================================================
   SETUP + DRAW
   ============================================================ */

void setup() {
  size(700, 366);
  minim = new Minim(this);
  loadSpeakers();
  loadCSV();
  loadSpeakerValues();
  loadAudio();
}

void draw() {
  background(30);
  if (!mousePressed) currentGestureTargetId = "";
  drawPlayer();
  drawSliders();
  saveSpeakerValuesIfChanged();
}


/* ============================================================
   DATASET LOADING
   ============================================================ */

void loadSpeakers() {
  File folder = new File(sketchPath(DATASET_PATH));
  speakers = folder.list((dir, name) -> new File(dir, name).isDirectory());
  Arrays.sort(speakers, new AlphanumComparator());

  speakerFiles = new String[speakers.length][];
  for (int i = 0; i < speakers.length; i++) {
    File sp = new File(sketchPath(DATASET_PATH + speakers[i]));
    speakerFiles[i] = sp.list((dir, name) -> name.toLowerCase().endsWith(".wav"));
    Arrays.sort(speakerFiles[i], new AlphanumComparator());
  }
}

void loadCSV() {
  File csvFile = new File(sketchPath(CSV_PATH));
  if (csvFile.exists()) {
    infoTable = loadTable(CSV_PATH, "header");
  } else {
    infoTable = new Table();
    infoTable.addColumn(COL_SPEAKER);
    infoTable.addColumn(COL_FEMININITY);
    infoTable.addColumn(COL_MASCULINITY);
    infoTable.addColumn(COL_ATYPICALITY);
    saveTable(infoTable, CSV_PATH);
  }
}


/* ============================================================
   AUDIO PLAYER UI
   ============================================================ */

void drawPlayer() {
  boolean markedUp = isMarkedUp();

  // ── Info row ─────────────────────────────────────────────
  fill(255);
  textSize(16);
  text("Speaker: " + speakers[speakerIndex], UI_LEFT, 20);
  text("File: "    + speakerFiles[speakerIndex][fileIndex], UI_LEFT, 45);

  fill(markedUp ? color(85, 255, 85) : color(255, 85, 85));
  text("Status: " + (markedUp ? "MARKED UP" : "NOT MARKED UP"), UI_LEFT, 70);

  // ── Navigation buttons ───────────────────────────────────
  drawNavRow( 96, "<<< Spk", -100, "Spk >>>", +100);
  drawNavRow(156, "<<  Spk",  -10, "Spk  >>", +10);

  // ── File controls ────────────────────────────────────────
  if (button(UI_LEFT,       216, 80, 40, "< File"))  switchFile(-1);
  if (button(UI_LEFT + 100, 216, 80, 40, "Play"))    togglePlay();
  if (button(UI_LEFT + 200, 216, 80, 40, "File >"))  switchFile(+1);

  // ── Speaker prev/restart/next (large) ────────────────────
  if (button(UI_LEFT,       276, 80, 80, "< Spk"))   switchSpeaker(-1);
  if (button(UI_LEFT + 100, 276, 80, 80, "Restart")) restartAudio();
  if (button(UI_LEFT + 200, 276, 80, 80, "Spk >"))   switchSpeaker(+1);
}

void drawNavRow(int y, String labelLeft, int dirLeft, String labelRight, int dirRight) {
  if (button(UI_LEFT,       y, 130, 40, labelLeft))  switchSpeaker(dirLeft);
  if (button(UI_LEFT + 150, y, 130, 40, labelRight)) switchSpeaker(dirRight);
}

void togglePlay() {
  if (player.isPlaying()) {
    player.pause();
  } else {
    if (player.position() == player.length()) player.rewind();
    player.play();
  }
}

void restartAudio() {
  player.rewind();
  player.play();
}


/* ============================================================
   SLIDERS UI
   ============================================================ */

void drawSliders() {
  fem  = labeledSlider(UI_RIGHT,  10, fem,  "FemininitySlider",  "Femininity");
  mas  = labeledSlider(UI_RIGHT,  98, mas,  "MasculinitySlider", "Masculinity");
  atyp = labeledSlider(UI_RIGHT, 186, atyp, "AtypicalitySlider", "Atypicality");

  // ── Preset buttons ───────────────────────────────────────
  if (button(310, 276, 80, 80, "F")) setPreset(1, 0, 0);
  if (button(410, 276, 80, 80, "M")) setPreset(0, 1, 0);
  if (button(510, 276, 80, 80, "A")) setPreset(0, 0, 1);
  if (button(610, 276, 80, 80, "S")) setPreset(0, 0, 0);
}

float labeledSlider(int x, int y, float val, String gestureId, String label) {
  val = slider(x, y, val, gestureId);
  fill(255);
  text(label, x + 10, y + 20);
  return val;
}

void setPreset(float f, float m, float a) {
  fem  = f;
  mas  = m;
  atyp = a;
  changed = true;
}


/* ============================================================
   SLIDER + BUTTON PRIMITIVES
   ============================================================ */

float slider(int x, int y, float val, String gestureId) {
  boolean hover   = mouseX > x && mouseX < x + SLIDER_W &&
                    mouseY > y && mouseY < y + SLIDER_H;
  boolean engaged = currentGestureTargetId.equals(gestureId);
  boolean hit     = mousePressed && hover;

  stroke(200);
  fill(hover ? 128 : 86);
  rect(x, y, SLIDER_W, SLIDER_H);

  fill(200, 100, 100);
  rect(x, y, val * SLIDER_W, SLIDER_H);

  if (hit || engaged) {
    currentGestureTargetId = gestureId;
    val = constrain((mouseX - x) / float(SLIDER_W), 0, 1);
    changed = true;
  }

  return val;
}

boolean button(int x, int y, int w, int h, String label) {
  boolean hover = mouseX > x && mouseX < x + w &&
                  mouseY > y && mouseY < y + h;
  fill(hover ? 128 : 86);
  rect(x, y, w, h);
  fill(255);
  text(label, x + 10, y + h / 2 + 5);
  return hover && mousePressed;
}


/* ============================================================
   SPEAKER + FILE NAVIGATION
   ============================================================ */

void switchFile(int dir) {
  fileIndex = (fileIndex + dir + speakerFiles[speakerIndex].length) %
               speakerFiles[speakerIndex].length;
  loadAudio();
}

void switchSpeaker(int dir) {
  saveSpeakerValuesIfChanged();
  speakerIndex = ((speakerIndex + dir) % speakers.length + speakers.length) % speakers.length;
  fileIndex = 0;
  loadSpeakerValues();
  loadAudio();
}


/* ============================================================
   CSV — LOAD + SAVE
   ============================================================ */

void loadSpeakerValues() {
  TableRow row = infoTable.findRow(speakers[speakerIndex], COL_SPEAKER);
  if (row == null) {
    fem = mas = atyp = 0;
  } else {
    fem  = row.getFloat(COL_FEMININITY);
    mas  = row.getFloat(COL_MASCULINITY);
    atyp = row.getFloat(COL_ATYPICALITY);
  }
  changed = false;
}

void saveSpeakerValuesIfChanged() {
  if (!changed) return;
  String sp = speakers[speakerIndex];
  TableRow row = infoTable.findRow(sp, COL_SPEAKER);
  if (row == null) {
    row = infoTable.addRow();
    row.setString(COL_SPEAKER, sp);
  }
  row.setFloat(COL_FEMININITY,  fem);
  row.setFloat(COL_MASCULINITY, mas);
  row.setFloat(COL_ATYPICALITY, atyp);
  saveTable(infoTable, CSV_PATH);
  changed = false;
}

boolean isMarkedUp() {
  return infoTable.findRow(speakers[speakerIndex], COL_SPEAKER) != null;
}


/* ============================================================
   AUDIO
   ============================================================ */

void loadAudio() {
  if (player != null) player.close();
  player = minim.loadFile(
    DATASET_PATH + speakers[speakerIndex] + "/" + speakerFiles[speakerIndex][fileIndex]
  );
}


/* ============================================================
   ALPHANUMERIC COMPARATOR
   ============================================================ */

class AlphanumComparator implements Comparator<String> {
  public int compare(String s1, String s2) {
    int i = 0, j = 0;
    while (i < s1.length() && j < s2.length()) {
      char c1 = s1.charAt(i), c2 = s2.charAt(j);
      if (Character.isDigit(c1) && Character.isDigit(c2)) {
        int start1 = i, start2 = j;
        while (i < s1.length() && Character.isDigit(s1.charAt(i))) i++;
        while (j < s2.length() && Character.isDigit(s2.charAt(j))) j++;
        String n1 = s1.substring(start1, i).replaceFirst("^0+", "");
        String n2 = s2.substring(start2, j).replaceFirst("^0+", "");
        if (n1.length() != n2.length()) return n1.length() - n2.length();
        int cmp = n1.compareTo(n2);
        if (cmp != 0) return cmp;
        continue;
      }
      int ci = Character.toLowerCase(c1), cj = Character.toLowerCase(c2);
      if (ci != cj) return ci - cj;
      i++; j++;
    }
    return (s1.length() - i) - (s2.length() - j);
  }
}
