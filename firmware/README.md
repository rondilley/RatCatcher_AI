# RatCatcher Status Panel Firmware

Firmware for the Elecrow CrowPanel ESP32 2.13-inch e-paper HMI display.
The panel shows the animals that RatCatcher found today. It also shows
the condition of the system. One USB cable connects the panel to the
host. That cable supplies the data and the electrical power.

## The screen

```
+-------------------------------------------+
| RatCatcher today               14:32  RUN |
| ----------------------------------------- |
|                   EYE     EAR     ALL     |
|  BIRDS             18       6      24     |
|  RODENTS            3       0       3     |
|  OTHER              1       0       1     |
| ----------------------------------------- |
| Last  Dark-eyed Junco          ear  14:29 |
| CAM 2  NPU ok  MIC on  47C  disk 61%  3d  |
+-------------------------------------------+
```

`EYE` is the count from the cameras. `EAR` is the count from the
microphones. The two counts are not connected. An animal that the
cameras see and the microphones hear adds one to each column. The video
path and the audio path do not compare their events.

The condition word shows `RUN` when the system operates correctly. It
shows `WARN` when a health check finds a problem. It shows `STOP` after
the host stops. It shows `STALE` when no data comes from the host for
seven minutes.

## Hardware

| Item | Value |
|---|---|
| Board | Elecrow CrowPanel ESP32 2.13-inch e-paper HMI |
| Processor | ESP32-S3, 8 MB flash, 8 MB PSRAM |
| Panel | 122 x 250, black and white, SSD1680Z or JD79661 |
| Link | USB Type-C, CH340 serial bridge, 115200 8N1 |

The firmware uses these pins:

| Signal | GPIO |
|---|---|
| Panel power | 7 |
| Power LED | 19 |
| SPI SCK, MOSI | 12, 11 |
| Panel RES, DC, CS, BUSY | 10, 13, 14, 9 |
| Buttons: home, exit, prev, next, ok | 2, 1, 6, 4, 5 |

The buttons are active low. Each button tells the host to send new data.
The panel has no menu.

## Build the firmware

```bash
# Compile only
./scripts/build_panel_firmware.sh

# Compile, then write the firmware to the panel
./scripts/build_panel_firmware.sh --upload --port /dev/ttyUSB0

# Delete the tools and start again
./scripts/build_panel_firmware.sh --clean
```

The script installs `arduino-cli`, the ESP32 core and ArduinoJson in
`firmware/.arduino/`. It installs no file in a different directory. The
first build downloads approximately 2.3 GB. It is necessary to wait some
minutes. Each build after the first one takes only seconds.

A correct build gives this result:

```
Sketch uses 340990 bytes (26%) of program storage space.
Global variables use 26952 bytes (8%) of dynamic memory.
```

The account must have write access to the serial port:

```bash
sudo usermod -aG dialout $USER
# Then log out and log in again
```

The ESP32 build tools use `python3` from the PATH. If a virtualenv for
a different processor is active, that `python3` cannot run, and the
build stops at the last step with `exec format error`. The build script
finds this condition and uses the system `python3`. But the correct
repair is to make `venv/` again on this machine.

## Files that this repository does not contain

The build script downloads the Elecrow e-paper driver into the sketch
directory:

```
EPD.h  EPD.cpp  EPD_Init.h  EPD_Init.cpp  EPDfont.h  spi.h  spi.cpp
```

These files are the property of Elecrow. Thus the script gets them from
the Elecrow repository. This repository does not contain a copy. This is
the same rule that applies to the BirdNET model, which
`scripts/download_models.sh` gets from Zenodo.

The firmware does not use GxEPD2. The board has one of two panel driver
chips: SSD1680Z or JD79661. The board revision gives the answer. GxEPD2
supports only the first chip. Thus a board with the second chip compiles
correctly but shows no data.

## Protocol

The host and the panel send one JSON object on each line. The text is
UTF-8. A newline character ends each line. The speed is 115200 baud.

A line protocol is better than a binary protocol here for two causes.
First, the panel discards a bad line at the next newline character.
Second, a person can read the data with a terminal and no decoder.

The host sends these messages:

```json
{"v":1,"t":"ping"}
{"v":1,"t":"status","clk":"14:32","st":"RUN","win":"today",
 "n":{"bird":[18,6],"rodent":[3,0],"other":[1,0]},
 "last":{"name":"Dark-eyed Junco","sense":"ear","clk":"14:29"},
 "sys":{"cam":2,"npu":1,"aud":1,"temp":47,"disk":61,"up":"3d"},
 "seq":7,"full":0}
```

The panel sends these messages:

```json
{"v":1,"t":"hello","panel":"crowpanel-2.13","fw":"1.0.0"}
{"v":1,"t":"ack","seq":7}
{"v":1,"t":"btn","id":"home"}
{"v":1,"t":"err","msg":"bad json"}
```

Each count is a pair. The first number is the count from the cameras.
The second number is the count from the microphones. The `full` field
tells the panel to do a full refresh and not a partial refresh.

The panel refuses each frame with an unknown version. It shows the
problem on the screen. An incorrect indication is worse than a panel
that shows no data.

The `hello` message lets the host find the correct port. The CrowPanel
has a standard CH340 descriptor. Many other boards have the same
descriptor. Thus the USB identifiers cannot identify the panel. The host
sends a ping to each possible port. Then it keeps the port that answers.

The host does not wait for an answer. A panel with a correct screen but
a defective transmit path continues to be usable. Thus the host
continues to send frames.

## Colour convention

The Elecrow driver gives the incorrect colour names at the pixel level.
`EPD_DrawPoint(x, y, BLACK)` sets the bit, and a set bit shows white.
The Elecrow code shows this. `EPD_ShowPicture` fills the two empty
columns on the right with `EPD_DrawPoint(248, row, BLACK)` to make an
empty margin. Thus:

- an empty page is `memset(ImageBW, 0xFF, ALLSCREEN_BYTES)`
- `EPD_ShowString(x, y, s, BLACK, size)` makes black text, as it reads
- `EPD_DrawLine(..., WHITE)` makes a black line, which it does not read

Read the note at the top of `ratcatcher_panel.ino` before you change a
function that makes an image.

## Refresh policy

A partial refresh is quiet and it operates quickly. But it keeps a ghost
of the previous image, and the ghosts increase. The host tells the panel
to do a full refresh after each 15 frames. The
`display.full_refresh_every` parameter sets this value. The firmware
does a full refresh after 40 continuous partial refreshes, whatever the
host tells it to do. Thus a host that does not tell it to do a full
refresh cannot let the ghosts increase without a limit.

The panel goes to sleep between refreshes. E-paper keeps its image with
no electrical power. An awake controller decreases the life of the
panel.

## Layout

The host sends numbers and not pixels. Thus a change to the numbers does
not make a new firmware necessary. But a change to the layout does.
Three files give one design. You must change all three together:

| File | Contents |
|---|---|
| `src/ratcatcher/display/protocol.py` | the data format |
| `src/ratcatcher/display/render.py` | the layout, and a text preview |
| `firmware/ratcatcher_panel/ratcatcher_panel.ino` | the layout, in pixels |

The `PIXEL_LAYOUT` value in `render.py` gives the pixel geometry next to
the character grid. Thus you can compare the two. To see the layout with
no panel connected, use this command:

```bash
ratcatcher display --preview
```
