/*
 * RatCatcher AI status panel
 * Elecrow CrowPanel ESP32 2.13-inch e-paper HMI display (122 x 250)
 *
 * Reads status frames from the USB serial port and draws them. One JSON
 * object per line at 115200 baud. The host sends numbers, not pixels, so
 * the link stays small and a person with a terminal program can read it.
 *
 * The protocol is defined in src/ratcatcher/display/protocol.py and the
 * layout in src/ratcatcher/display/render.py. Those two files and this
 * one are three statements of one design. Change them together.
 *
 * Build and flash with scripts/build_panel_firmware.sh, which fetches
 * the Elecrow EPD driver files into this directory first. They are not
 * kept in this repository.
 *
 * COLOUR CONVENTION -- read this before changing any drawing call.
 * The Elecrow driver names its colours backwards at the pixel level.
 * EPD_DrawPoint(x, y, BLACK) SETS the bit, and a set bit shows WHITE.
 * The vendor's own code proves it: EPD_ShowPicture fills the two unused
 * right-hand columns with EPD_DrawPoint(248, row, BLACK) to leave a
 * blank margin. So:
 *   - a blank page is memset(ImageBW, 0xFF, ...), every bit set
 *   - EPD_ShowString(x, y, s, BLACK, size) draws BLACK text, as intended
 *   - EPD_DrawLine(..., WHITE) draws a BLACK line
 * The text call reads correctly and the line call reads backwards. Both
 * are right.
 */

#include <ArduinoJson.h>
#include "EPD.h"

extern uint8_t ImageBW[ALLSCREEN_BYTES];

/* ------------------------------------------------------------------ */
/* Board                                                               */
/* ------------------------------------------------------------------ */

#define PIN_POWER_LED 19
#define PIN_EPD_POWER 7

/* Buttons are wired active low. */
#define KEY_HOME 2
#define KEY_EXIT 1
#define KEY_PREV 6
#define KEY_NEXT 4
#define KEY_OK   5

static const uint8_t KEY_PINS[] = {KEY_HOME, KEY_EXIT, KEY_PREV, KEY_NEXT, KEY_OK};
static const char *KEY_NAMES[] = {"home", "exit", "prev", "next", "ok"};
static const size_t KEY_COUNT = sizeof(KEY_PINS) / sizeof(KEY_PINS[0]);

/* ------------------------------------------------------------------ */
/* Protocol                                                            */
/* ------------------------------------------------------------------ */

#define PROTOCOL_VERSION 1
#define PANEL_MODEL "crowpanel-2.13"
#define FIRMWARE_VERSION "1.0.0"

/* Longer than the host's 512-byte limit, so a legal frame always fits
 * and an over-long line is discarded whole rather than parsed in half. */
#define RX_BUFFER_BYTES 640

/* The host sends a heartbeat every 300 seconds. Past this, the numbers
 * on the screen are no longer known to be current and the header says
 * so. A panel that keeps reporting RUN for a host that has died is
 * worse than a blank one. */
#define HOST_STALE_MS 420000UL

/* ------------------------------------------------------------------ */
/* Layout, in pixels on the 248 x 122 landscape frame                  */
/* ------------------------------------------------------------------ */

#define FONT_BIG   16
#define FONT_SMALL 12

#define X_LABEL      2
#define X_WINDOW    90
#define X_RIGHT_EDGE 246

#define Y_HEADER      1
#define Y_RULE_TOP   19
#define Y_COLHEAD     21
#define Y_ROW_1       35
#define Y_ROW_2       53
#define Y_ROW_3       71
#define Y_RULE_BOTTOM 89
#define Y_FOOTER_1    92
#define Y_FOOTER_2   107

#define COL_EYE 130
#define COL_EAR 180
#define COL_ALL 238

#define X_CLOCK_RIGHT 200

/* ------------------------------------------------------------------ */
/* State                                                               */
/* ------------------------------------------------------------------ */

struct PanelState {
  char clock[8];
  char state[8];
  char window[8];
  long seen[3];
  long heard[3];
  char lastName[24];
  char lastSense[4];
  char lastClock[8];
  bool hasLast;
  long cameras;
  bool npu;
  bool audio;
  bool hasTemp;
  long tempC;
  bool hasDisk;
  long diskPct;
  char uptime[8];
};

static PanelState g_state;
static bool g_haveFrame = false;
static bool g_stale = false;
static unsigned long g_lastFrameMs = 0;
static uint16_t g_partialCount = 0;

static char g_rx[RX_BUFFER_BYTES];
static size_t g_rxLen = 0;
static bool g_rxOverflow = false;

static bool g_keyDown[5] = {false, false, false, false, false};
static unsigned long g_keyChangedMs[5] = {0, 0, 0, 0, 0};

/* ------------------------------------------------------------------ */
/* Drawing helpers                                                     */
/* ------------------------------------------------------------------ */

static void clearPage() {
  /* Every bit set is a white page. See the colour note at the top. */
  memset(ImageBW, 0xFF, ALLSCREEN_BYTES);
}

static void drawText(uint16_t x, uint16_t y, const char *text, uint8_t size) {
  EPD_ShowString(x, y, text, BLACK, size);
}

static void drawTextRight(uint16_t rightEdge, uint16_t y, const char *text, uint8_t size) {
  uint16_t width = (uint16_t)strlen(text) * (size / 2);
  uint16_t x = (rightEdge > width) ? (rightEdge - width) : 0;
  EPD_ShowString(x, y, text, BLACK, size);
}

static void drawRule(uint16_t y) {
  /* WHITE draws black here. See the colour note at the top. */
  EPD_DrawLine(X_LABEL, y, X_RIGHT_EDGE, y, WHITE);
}

static void drawNumberRight(uint16_t rightEdge, uint16_t y, long value, uint8_t size) {
  char text[12];
  snprintf(text, sizeof(text), "%ld", value);
  drawTextRight(rightEdge, y, text, size);
}

/* ------------------------------------------------------------------ */
/* Screens                                                             */
/* ------------------------------------------------------------------ */

static void drawSplash(const char *message) {
  clearPage();
  drawText(X_LABEL, Y_HEADER, "RatCatcher AI", FONT_BIG);
  drawRule(Y_RULE_TOP);
  drawText(X_LABEL, Y_ROW_1, message, FONT_BIG);
  drawText(X_LABEL, Y_FOOTER_1, "Panel " PANEL_MODEL, FONT_SMALL);
  drawText(X_LABEL, Y_FOOTER_2, "Firmware " FIRMWARE_VERSION "  115200 8N1", FONT_SMALL);
}

static void drawStatus() {
  static const char *ROW_LABELS[3] = {"BIRDS", "RODENTS", "OTHER"};
  static const uint16_t ROW_Y[3] = {Y_ROW_1, Y_ROW_2, Y_ROW_3};

  clearPage();

  drawText(X_LABEL, Y_HEADER, "RatCatcher", FONT_BIG);
  drawText(X_WINDOW, Y_HEADER, g_state.window, FONT_BIG);
  drawTextRight(X_CLOCK_RIGHT, Y_HEADER, g_state.clock, FONT_BIG);
  drawTextRight(X_RIGHT_EDGE, Y_HEADER, g_stale ? "STALE" : g_state.state, FONT_BIG);

  drawRule(Y_RULE_TOP);

  drawTextRight(COL_EYE, Y_COLHEAD, "EYE", FONT_SMALL);
  drawTextRight(COL_EAR, Y_COLHEAD, "EAR", FONT_SMALL);
  drawTextRight(COL_ALL, Y_COLHEAD, "ALL", FONT_SMALL);

  for (int row = 0; row < 3; row++) {
    drawText(X_LABEL, ROW_Y[row], ROW_LABELS[row], FONT_BIG);
    drawNumberRight(COL_EYE, ROW_Y[row], g_state.seen[row], FONT_BIG);
    drawNumberRight(COL_EAR, ROW_Y[row], g_state.heard[row], FONT_BIG);
    drawNumberRight(COL_ALL, ROW_Y[row], g_state.seen[row] + g_state.heard[row], FONT_BIG);
  }

  drawRule(Y_RULE_BOTTOM);

  if (g_state.hasLast) {
    char head[40];
    char tail[16];
    snprintf(head, sizeof(head), "Last  %s", g_state.lastName);
    snprintf(tail, sizeof(tail), "%s  %s", g_state.lastSense, g_state.lastClock);
    drawText(X_LABEL, Y_FOOTER_1, head, FONT_SMALL);
    drawTextRight(X_RIGHT_EDGE, Y_FOOTER_1, tail, FONT_SMALL);
  } else {
    drawText(X_LABEL, Y_FOOTER_1, "Last  (nothing yet)", FONT_SMALL);
  }

  /* Worst case is "CAM 10  NPU ok  MIC on  100C  disk 100%  up 365d",
   * 47 characters. 48 bytes would leave no headroom at all, and the
   * host controls every one of those values. At the 6-pixel font the
   * screen fits 41 characters, so a longer line is cut by the bounds
   * check in EPD_ShowString rather than by this buffer. */
  char footer[64];
  int used = snprintf(footer, sizeof(footer), "CAM %ld  NPU %s  MIC %s",
                      g_state.cameras,
                      g_state.npu ? "ok" : "--",
                      g_state.audio ? "on" : "--");
  if (g_state.hasTemp && used > 0 && used < (int)sizeof(footer)) {
    used += snprintf(footer + used, sizeof(footer) - used, "  %ldC", g_state.tempC);
  }
  if (g_state.hasDisk && used > 0 && used < (int)sizeof(footer)) {
    used += snprintf(footer + used, sizeof(footer) - used, "  disk %ld%%", g_state.diskPct);
  }
  if (used > 0 && used < (int)sizeof(footer)) {
    snprintf(footer + used, sizeof(footer) - used, "  up %s", g_state.uptime);
  }
  drawText(X_LABEL, Y_FOOTER_2, footer, FONT_SMALL);
}

/* ------------------------------------------------------------------ */
/* Panel refresh                                                       */
/* ------------------------------------------------------------------ */

/* Upper bound on consecutive partial refreshes. The host asks for a
 * full refresh on its own schedule; this is the backstop for a host
 * that never does, so ghosting cannot build up without limit. */
#define MAX_PARTIAL_REFRESHES 40

static void pushToPanel(bool full) {
  if (g_partialCount >= MAX_PARTIAL_REFRESHES) {
    full = true;
  }

  if (full) {
    EPD_Init();
  } else {
    EPD_HW_Init_Fast();
  }

  EPD_DisplayImage(ImageBW);

  if (full) {
    EPD_Update();
    g_partialCount = 0;
  } else {
    EPD_Update_Fast();
    g_partialCount++;
  }

  /* Sleep between refreshes. The panel holds its image with no power,
   * and leaving the controller awake is what shortens its life. */
  EPD_Sleep();
}

/* ------------------------------------------------------------------ */
/* Serial                                                              */
/* ------------------------------------------------------------------ */

static void sendHello() {
  Serial.print("{\"v\":");
  Serial.print(PROTOCOL_VERSION);
  Serial.print(",\"t\":\"hello\",\"panel\":\"" PANEL_MODEL "\",\"fw\":\"" FIRMWARE_VERSION "\"}\n");
}

static void sendAck(long seq) {
  Serial.print("{\"v\":");
  Serial.print(PROTOCOL_VERSION);
  Serial.print(",\"t\":\"ack\",\"seq\":");
  Serial.print(seq);
  Serial.print("}\n");
}

static void sendError(const char *message) {
  Serial.print("{\"v\":");
  Serial.print(PROTOCOL_VERSION);
  Serial.print(",\"t\":\"err\",\"msg\":\"");
  Serial.print(message);
  Serial.print("\"}\n");
}

static void sendButton(const char *name) {
  Serial.print("{\"v\":");
  Serial.print(PROTOCOL_VERSION);
  Serial.print(",\"t\":\"btn\",\"id\":\"");
  Serial.print(name);
  Serial.print("\"}\n");
}

static void copyString(char *destination, size_t size, const char *source) {
  if (source == NULL) {
    destination[0] = '\0';
    return;
  }
  strncpy(destination, source, size - 1);
  destination[size - 1] = '\0';
}

static void readCounts(JsonObject counts, const char *key, int row) {
  g_state.seen[row] = 0;
  g_state.heard[row] = 0;
  JsonArray pair = counts[key].as<JsonArray>();
  if (pair.isNull() || pair.size() < 2) {
    return;
  }
  g_state.seen[row] = pair[0].as<long>();
  g_state.heard[row] = pair[1].as<long>();
}

static void applyStatus(JsonDocument &doc) {
  copyString(g_state.clock, sizeof(g_state.clock), doc["clk"] | "--:--");
  copyString(g_state.state, sizeof(g_state.state), doc["st"] | "?");
  copyString(g_state.window, sizeof(g_state.window), doc["win"] | "");

  JsonObject counts = doc["n"].as<JsonObject>();
  if (!counts.isNull()) {
    readCounts(counts, "bird", 0);
    readCounts(counts, "rodent", 1);
    readCounts(counts, "other", 2);
  }

  JsonObject last = doc["last"].as<JsonObject>();
  g_state.hasLast = !last.isNull();
  if (g_state.hasLast) {
    copyString(g_state.lastName, sizeof(g_state.lastName), last["name"] | "");
    copyString(g_state.lastSense, sizeof(g_state.lastSense), last["sense"] | "");
    copyString(g_state.lastClock, sizeof(g_state.lastClock), last["clk"] | "");
  }

  JsonObject system = doc["sys"].as<JsonObject>();
  if (!system.isNull()) {
    g_state.cameras = system["cam"] | 0;
    g_state.npu = (system["npu"] | 0) != 0;
    g_state.audio = (system["aud"] | 0) != 0;
    g_state.hasTemp = !system["temp"].isNull();
    g_state.tempC = system["temp"] | 0;
    g_state.hasDisk = !system["disk"].isNull();
    g_state.diskPct = system["disk"] | 0;
    copyString(g_state.uptime, sizeof(g_state.uptime), system["up"] | "?");
  }

  g_haveFrame = true;
  g_stale = false;
  g_lastFrameMs = millis();
}

static void handleLine(const char *line) {
  JsonDocument doc;
  DeserializationError error = deserializeJson(doc, line);
  if (error) {
    sendError("bad json");
    return;
  }

  long version = doc["v"] | 0;
  if (version != PROTOCOL_VERSION) {
    /* Show the mismatch rather than drawing numbers this firmware may
     * have misread. A wrong reading is worse than a stopped panel. */
    sendError("protocol version");
    drawSplash("HOST PROTOCOL MISMATCH");
    pushToPanel(true);
    return;
  }

  const char *type = doc["t"] | "";

  if (strcmp(type, "ping") == 0) {
    sendHello();
    return;
  }

  if (strcmp(type, "status") != 0) {
    return;
  }

  applyStatus(doc);
  drawStatus();
  pushToPanel((doc["full"] | 0) != 0);
  sendAck(doc["seq"] | 0);
}

static void pumpSerial() {
  while (Serial.available() > 0) {
    int value = Serial.read();
    if (value < 0) {
      break;
    }
    char character = (char)value;

    if (character == '\n' || character == '\r') {
      if (g_rxOverflow) {
        /* The line was longer than the buffer. It was never complete,
         * so discard all of it instead of parsing the front half. */
        sendError("line too long");
        g_rxOverflow = false;
        g_rxLen = 0;
        continue;
      }
      if (g_rxLen > 0) {
        g_rx[g_rxLen] = '\0';
        handleLine(g_rx);
        g_rxLen = 0;
      }
      continue;
    }

    if (g_rxLen + 1 >= RX_BUFFER_BYTES) {
      g_rxOverflow = true;
      continue;
    }
    g_rx[g_rxLen++] = character;
  }
}

/* ------------------------------------------------------------------ */
/* Buttons                                                             */
/* ------------------------------------------------------------------ */

static void pumpButtons() {
  unsigned long nowMs = millis();
  for (size_t index = 0; index < KEY_COUNT; index++) {
    bool down = digitalRead(KEY_PINS[index]) == LOW;
    if (down == g_keyDown[index]) {
      continue;
    }
    if (nowMs - g_keyChangedMs[index] < 40) {
      continue;  /* contact bounce */
    }
    g_keyChangedMs[index] = nowMs;
    g_keyDown[index] = down;
    if (down) {
      sendButton(KEY_NAMES[index]);
    }
  }
}

/* ------------------------------------------------------------------ */
/* Stale host                                                          */
/* ------------------------------------------------------------------ */

static void checkStale() {
  if (!g_haveFrame || g_stale) {
    return;
  }
  if (millis() - g_lastFrameMs < HOST_STALE_MS) {
    return;
  }
  g_stale = true;
  drawStatus();
  pushToPanel(true);
}

/* ------------------------------------------------------------------ */
/* Entry points                                                        */
/* ------------------------------------------------------------------ */

void setup() {
  Serial.begin(115200);

  pinMode(PIN_POWER_LED, OUTPUT);
  digitalWrite(PIN_POWER_LED, HIGH);
  pinMode(PIN_EPD_POWER, OUTPUT);
  digitalWrite(PIN_EPD_POWER, HIGH);

  for (size_t index = 0; index < KEY_COUNT; index++) {
    pinMode(KEY_PINS[index], INPUT);
  }

  memset(&g_state, 0, sizeof(g_state));
  copyString(g_state.clock, sizeof(g_state.clock), "--:--");
  copyString(g_state.state, sizeof(g_state.state), "WAIT");
  copyString(g_state.uptime, sizeof(g_state.uptime), "?");

  EPD_GPIOInit();
  EPD_Init();
  EPD_Clear();
  EPD_Update();

  drawSplash("Waiting for host");
  pushToPanel(true);

  sendHello();
}

void loop() {
  pumpSerial();
  pumpButtons();
  checkStale();
  delay(5);
}
