const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const path = require("path");
const {
  FaCamera, FaBug, FaRobot, FaCube, FaExclamationTriangle,
  FaLightbulb, FaBalanceScale, FaClock, FaMapMarkerAlt,
  FaCut, FaCheck, FaTimes, FaArrowRight, FaBrain,
} = require("react-icons/fa");

function renderIconSvg(IconComponent, color = "#000000", size = 256) {
  return ReactDOMServer.renderToStaticMarkup(
    React.createElement(IconComponent, { color, size: String(size) })
  );
}

async function iconToBase64Png(IconComponent, color, size = 256) {
  const svg = renderIconSvg(IconComponent, color, size);
  const pngBuffer = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + pngBuffer.toString("base64");
}

const C = {
  cream: "F7F1E8",
  ink: "2C2416",
  muted: "6B5849",
  white: "FFFFFF",
  teal: "2A9D8F",
  tealDark: "1F7A6E",
  coral: "E76F51",
  navy: "264653",
  gold: "E9C46A",
  sage: "8AB17D",
  sky: "A8DADC",
  card: "FFFBF5",
};

const FONT_H = "Trebuchet MS";
const FONT_B = "Calibri";
const W = 13.333;
const H = 7.5;
const ASSETS = path.join(__dirname, "deck-assets");

const sh = () => ({ type: "outer", blur: 6, offset: 2, angle: 135, color: "000000", opacity: 0.1 });

async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";
  pres.title = "Can a fly brain pick up a cube?";
  pres.author = "fly-brain";
  pres.subject = "Teen explainer: problem, two idea lists, recommended next step";

  const icons = {};
  const imap = {
    cam: [FaCamera, "#" + C.white],
    bug: [FaBug, "#" + C.white],
    robot: [FaRobot, "#" + C.white],
    cube: [FaCube, "#" + C.white],
    warn: [FaExclamationTriangle, "#" + C.white],
    bulb: [FaLightbulb, "#" + C.white],
    scale: [FaBalanceScale, "#" + C.white],
    clock: [FaClock, "#" + C.white],
    pin: [FaMapMarkerAlt, "#" + C.white],
    cut: [FaCut, "#" + C.white],
    check: [FaCheck, "#" + C.white],
    times: [FaTimes, "#" + C.white],
    arrow: [FaArrowRight, "#" + C.white],
    brain: [FaBrain, "#" + C.white],
  };
  for (const [k, [Ic, col]] of Object.entries(imap)) {
    icons[k] = await iconToBase64Png(Ic, col, 256);
  }

  function header(slide, section) {
    slide.addText("FLY BRAIN  ·  CUBE PICK", {
      x: 0.55, y: 0.28, w: 7, h: 0.28,
      fontFace: FONT_H, fontSize: 11, bold: true, color: C.muted, charSpacing: 2, margin: 0,
    });
    slide.addText(section, {
      x: 7.5, y: 0.28, w: 5.3, h: 0.28,
      fontFace: FONT_H, fontSize: 11, bold: true, color: C.muted, align: "right", margin: 0,
    });
  }

  function circleIcon(slide, x, y, fill, iconKey) {
    slide.addShape(pres.shapes.OVAL, {
      x, y, w: 0.42, h: 0.42, fill: { color: fill }, line: { type: "none" },
    });
    slide.addImage({ data: icons[iconKey], x: x + 0.09, y: y + 0.09, w: 0.24, h: 0.24 });
  }

  // ----- 1 Title -----
  {
    const s = pres.addSlide();
    s.background = { color: C.navy };
    s.addShape(pres.shapes.OVAL, {
      x: -2.4, y: -2.2, w: 4.2, h: 4.2, fill: { color: "1A333C" }, line: { type: "none" },
    });
    s.addText("Can a tiny fly brain\npick up a cube?", {
      x: 0.7, y: 1.5, w: 7.2, h: 2.2,
      fontFace: FONT_H, fontSize: 38, bold: true, color: C.white, margin: 0,
    });
    s.addText("A simple story about cameras, fake fly brains, and a robot arm.\nWritten so a 14-year-old can follow every step.", {
      x: 0.7, y: 3.85, w: 7, h: 1.1,
      fontFace: FONT_B, fontSize: 18, color: C.sky, margin: 0,
    });
    s.addImage({
      path: path.join(ASSETS, "table-arm.jpg"),
      x: 8.15, y: 1.35, w: 4.55, h: 4.55 * (9 / 16),
      rounding: true,
    });
  }

  // ----- 2 Goal -----
  {
    const s = pres.addSlide();
    s.background = { color: C.cream };
    header(s, "THE GOAL");
    s.addText("We want this chain to work", {
      x: 0.55, y: 0.65, w: 12, h: 0.55,
      fontFace: FONT_H, fontSize: 28, bold: true, color: C.ink, margin: 0,
    });
    const steps = [
      [C.teal, "cam", "1. Look", "A camera takes a photo of the cube on the table."],
      [C.navy, "brain", "2. Think", "A mini copy of a fly brain turns the photo into signals."],
      [C.coral, "robot", "3. Move", "Those signals tell the robot arm how far to go."],
      [C.gold, "cube", "4. Pick", "The cube lifts. That is the only real win."],
    ];
    steps.forEach((st, i) => {
      const x = 0.55 + i * 3.2;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x, y: 1.45, w: 3.0, h: 4.5, fill: { color: C.card }, rectRadius: 0.12,
        shadow: sh(), line: { type: "none" },
      });
      s.addShape(pres.shapes.OVAL, {
        x: x + 1.05, y: 1.75, w: 0.9, h: 0.9, fill: { color: st[0] }, line: { type: "none" },
      });
      s.addImage({ data: icons[st[1]], x: x + 1.26, y: 1.96, w: 0.48, h: 0.48 });
      s.addText(st[2], {
        x: x + 0.18, y: 2.85, w: 2.64, h: 0.5,
        fontFace: FONT_H, fontSize: 20, bold: true, color: C.ink, align: "center", margin: 0,
      });
      s.addText(st[3], {
        x: x + 0.22, y: 3.45, w: 2.56, h: 2.0,
        fontFace: FONT_B, fontSize: 15, color: C.muted, align: "center", margin: 0,
      });
    });
  }

  // ----- 3 Pipeline infographic -----
  {
    const s = pres.addSlide();
    s.background = { color: C.cream };
    header(s, "HOW IT IS SUPPOSED TO WORK");
    s.addText("Photo in. Millimetres out.", {
      x: 0.55, y: 0.65, w: 12, h: 0.5,
      fontFace: FONT_H, fontSize: 28, bold: true, color: C.ink, margin: 0,
    });
    const pipe = [
      [C.teal, "cam", "Camera", "Sees the table"],
      [C.navy, "brain", "Mini brain", "About 60,000 cells"],
      [C.coral, "bug", "Output wires", "Left, right, down…"],
      [C.gold, "robot", "Robot math", "Turns that into a move"],
    ];
    pipe.forEach((p, i) => {
      const x = 0.55 + i * 3.2;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x, y: 1.4, w: 2.95, h: 2.35, fill: { color: p[0] }, rectRadius: 0.1, line: { type: "none" },
      });
      s.addImage({ data: icons[p[1]], x: x + 1.15, y: 1.58, w: 0.55, h: 0.55 });
      s.addText(p[2], {
        x: x + 0.12, y: 2.25, w: 2.7, h: 0.45,
        fontFace: FONT_H, fontSize: 18, bold: true, color: C.white, align: "center", margin: 0,
      });
      s.addText(p[3], {
        x: x + 0.12, y: 2.7, w: 2.7, h: 0.7,
        fontFace: FONT_B, fontSize: 14, color: C.sky, align: "center", margin: 0,
      });
      if (i < 3) {
        s.addShape(pres.shapes.RIGHT_ARROW, {
          x: x + 2.85, y: 2.35, w: 0.38, h: 0.28,
          fill: { color: C.ink }, line: { type: "none" },
        });
      }
    });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 0.55, y: 4.05, w: 12.25, h: 2.7, fill: { color: C.card }, rectRadius: 0.1,
      shadow: sh(), line: { type: "none" },
    });
    s.addText("Kid translation", {
      x: 0.8, y: 4.25, w: 11.8, h: 0.4,
      fontFace: FONT_H, fontSize: 16, bold: true, color: C.teal, margin: 0,
    });
    s.addText("A real fruit fly has special output cells that say things like “go left” or “go down.” We copied a slice of that wiring into a computer. The robot is not the fly. The computer slice is. If the slice never shouts “down,” the arm will not pick the cube — no matter how smart the rest of the code looks.", {
      x: 0.8, y: 4.75, w: 11.8, h: 1.7,
      fontFace: FONT_B, fontSize: 16, color: C.ink, margin: 0,
    });
  }

  // ----- 4 Problem -----
  {
    const s = pres.addSlide();
    s.background = { color: C.cream };
    header(s, "THE PROBLEM");
    s.addText("The “down” shout is missing or wrong", {
      x: 0.55, y: 0.65, w: 12, h: 0.5,
      fontFace: FONT_H, fontSize: 28, bold: true, color: C.ink, margin: 0,
    });
    // two columns
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 0.55, y: 1.35, w: 6.0, h: 5.4, fill: { color: C.card }, rectRadius: 0.1, shadow: sh(), line: { type: "none" },
    });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 6.8, y: 1.35, w: 6.0, h: 5.4, fill: { color: C.card }, rectRadius: 0.1, shadow: sh(), line: { type: "none" },
    });
    s.addText("What a human demo does", {
      x: 0.8, y: 1.55, w: 5.5, h: 0.45, fontFace: FONT_H, fontSize: 18, bold: true, color: C.teal, margin: 0,
    });
    s.addText("Someone already taught the same robot to pick this cube by copying human moves. In one second the arm can drop about 50 millimetres — a real reach down.", {
      x: 0.8, y: 2.1, w: 5.5, h: 1.5, fontFace: FONT_B, fontSize: 16, color: C.ink, margin: 0,
    });
    s.addText("−50 mm", {
      x: 0.8, y: 3.7, w: 5.5, h: 0.9, fontFace: FONT_H, fontSize: 40, bold: true, color: C.teal, margin: 0,
    });
    s.addText("down, in one second of a real pick", {
      x: 0.8, y: 4.6, w: 5.5, h: 0.5, fontFace: FONT_B, fontSize: 15, color: C.muted, margin: 0,
    });
    s.addText("What the fly slice says", {
      x: 7.05, y: 1.55, w: 5.5, h: 0.45, fontFace: FONT_H, fontSize: 18, bold: true, color: C.coral, margin: 0,
    });
    s.addText("On the same photos, the fly slice mostly whispers “a bit to the side.” The “go down” wires are quiet, late, or pointed the wrong way.", {
      x: 7.05, y: 2.1, w: 5.5, h: 1.5, fontFace: FONT_B, fontSize: 16, color: C.ink, margin: 0,
    });
    s.addText("−1 mm", {
      x: 7.05, y: 3.7, w: 5.5, h: 0.9, fontFace: FONT_H, fontSize: 40, bold: true, color: C.coral, margin: 0,
    });
    s.addText("and sometimes the opposite of the real move", {
      x: 7.05, y: 4.6, w: 5.5, h: 0.5, fontFace: FONT_B, fontSize: 15, color: C.muted, margin: 0,
    });
  }

  // ----- 5 Tug of war -----
  {
    const s = pres.addSlide();
    s.background = { color: C.cream };
    header(s, "WHY “TRY HARDER” BREAKS");
    s.addText("A tug of war inside the mini brain", {
      x: 0.55, y: 0.65, w: 12, h: 0.5,
      fontFace: FONT_H, fontSize: 28, bold: true, color: C.ink, margin: 0,
    });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 0.55, y: 1.4, w: 5.9, h: 2.4, fill: { color: C.teal }, rectRadius: 0.1, line: { type: "none" },
    });
    s.addText("Wait longer", {
      x: 0.8, y: 1.6, w: 5.4, h: 0.45, fontFace: FONT_H, fontSize: 22, bold: true, color: C.white, margin: 0,
    });
    s.addText("The “down” cells wake up late. If we let the fake brain run longer, they finally shout. That sounds good.", {
      x: 0.8, y: 2.15, w: 5.4, h: 1.3, fontFace: FONT_B, fontSize: 16, color: C.white, margin: 0,
    });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 6.9, y: 1.4, w: 5.9, h: 2.4, fill: { color: C.coral }, rectRadius: 0.1, line: { type: "none" },
    });
    s.addText("The stop alarm also wakes up", {
      x: 7.15, y: 1.6, w: 5.4, h: 0.45, fontFace: FONT_H, fontSize: 22, bold: true, color: C.white, margin: 0,
    });
    s.addText("The same extra time turns on a “stop!” cell. Then we are not allowed to trust the move. We lose safety if we ignore it.", {
      x: 7.15, y: 2.15, w: 5.4, h: 1.3, fontFace: FONT_B, fontSize: 16, color: C.white, margin: 0,
    });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 0.55, y: 4.1, w: 12.25, h: 2.65, fill: { color: C.navy }, rectRadius: 0.1, line: { type: "none" },
    });
    s.addText("Same trap if we add the whole fly brain", {
      x: 0.8, y: 4.3, w: 11.8, h: 0.45, fontFace: FONT_H, fontSize: 20, bold: true, color: C.gold, margin: 0,
    });
    s.addText("The live robot must stay on a small slice (about 60,000 cells), not all 166,000. Growing the slice helps the missing wires — and makes the computer too slow to drive the arm. Win one thing, lose the other. That is the knot we have to untie, not “average it.”", {
      x: 0.8, y: 4.85, w: 11.8, h: 1.55, fontFace: FONT_B, fontSize: 16, color: C.white, margin: 0,
    });
  }

  // ----- 6 First five -----
  {
    const s = pres.addSlide();
    s.background = { color: C.cream };
    header(s, "SET A  ·  FIRST FIVE IDEAS");
    s.addText("The “try harder” list", {
      x: 0.55, y: 0.62, w: 12, h: 0.45,
      fontFace: FONT_H, fontSize: 26, bold: true, color: C.ink, margin: 0,
    });
    const ideas = [
      ["1", "Put missing cells back", "Add the helper cells we cut out, so the “left / down” wires have friends again."],
      ["2", "Turn some volumes down", "Quiet the stop-alarm cells so the late “down” cells can speak."],
      ["3", "Two clocks", "Read the move early, read the stop alarm later. Do not pick one waiting time."],
      ["4", "Listen to whoever is talking", "Use the wires that already change on pick photos. Stop pretending silent wires are joysticks."],
      ["5", "Aim the photo better", "Shine the picture onto the right spots in the mini brain, not as one big smear."],
    ];
    ideas.forEach((it, i) => {
      const y = 1.18 + i * 1.1;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x: 0.55, y, w: 12.25, h: 1.0, fill: { color: C.card }, rectRadius: 0.08, shadow: sh(), line: { type: "none" },
      });
      s.addShape(pres.shapes.OVAL, {
        x: 0.75, y: y + 0.22, w: 0.56, h: 0.56, fill: { color: C.navy }, line: { type: "none" },
      });
      s.addText(it[0], {
        x: 0.75, y: y + 0.28, w: 0.56, h: 0.44,
        fontFace: FONT_H, fontSize: 18, bold: true, color: C.white, align: "center", margin: 0,
      });
      s.addText(it[1], {
        x: 1.5, y: y + 0.12, w: 10.9, h: 0.38,
        fontFace: FONT_H, fontSize: 16, bold: true, color: C.ink, margin: 0,
      });
      s.addText(it[2], {
        x: 1.5, y: y + 0.5, w: 10.9, h: 0.4,
        fontFace: FONT_B, fontSize: 14, color: C.muted, margin: 0,
      });
    });
  }

  // ----- 7 Why weak -----
  {
    const s = pres.addSlide();
    s.background = { color: C.cream };
    header(s, "SET A  ·  HONEST LOOK");
    s.addText("These ideas are mostly the same habit", {
      x: 0.55, y: 0.65, w: 12, h: 0.5,
      fontFace: FONT_H, fontSize: 28, bold: true, color: C.ink, margin: 0,
    });
    s.addImage({
      path: path.join(ASSETS, "fly-brain.jpg"),
      x: 0.55, y: 1.4, w: 5.8, h: 5.8 * (9 / 16),
    });
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 6.7, y: 1.4, w: 6.1, h: 5.3, fill: { color: C.navy }, rectRadius: 0.1, line: { type: "none" },
    });
    s.addText("If your bike chain is off, you can push harder on the pedals. You go nowhere faster.\n\nBigger brain, more waiting, louder volume — same chain. We already tried “a bit more waiting” and “a bit more volume.” The stop alarm got louder too.\n\nSo we asked a different question: how do you win without paying?", {
      x: 6.95, y: 1.7, w: 5.6, h: 4.7,
      fontFace: FONT_B, fontSize: 17, color: C.white, margin: 0,
    });
  }

  // ----- 8 TRIZ kid -----
  {
    const s = pres.addSlide();
    s.background = { color: C.cream };
    header(s, "A BETTER QUESTION");
    s.addText("Win. Do not pay.", {
      x: 0.55, y: 0.62, w: 12, h: 0.5,
      fontFace: FONT_H, fontSize: 28, bold: true, color: C.ink, margin: 0,
    });
    const boxes = [
      [C.teal, "The knot", "If we add missing cells, the live robot gets too slow. If we wait longer, the stop alarm screams. A middle wait is a shrug, not a fix."],
      [C.navy, "The dream result", "The photo already knows “down.” We should not need a giant extra brain, a fake map, or a human teacher on every tick."],
      [C.coral, "The old habit", "Keep twisting the same eight output wires. That is like tying a longer rope when the rope was the wrong tool."],
    ];
    boxes.forEach((b, i) => {
      const x = 0.55 + i * 4.2;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x, y: 1.35, w: 4.0, h: 5.35, fill: { color: b[0] }, rectRadius: 0.1, line: { type: "none" },
      });
      s.addText(b[1], {
        x: x + 0.25, y: 1.6, w: 3.5, h: 0.7,
        fontFace: FONT_H, fontSize: 22, bold: true, color: C.white, margin: 0,
      });
      s.addText(b[2], {
        x: x + 0.25, y: 2.45, w: 3.5, h: 3.8,
        fontFace: FONT_B, fontSize: 16, color: C.white, margin: 0,
      });
    });
  }

  // ----- 9 Second five -----
  {
    const s = pres.addSlide();
    s.background = { color: C.cream };
    header(s, "SET B  ·  AFTER THAT QUESTION");
    s.addText("Five ideas that try not to pay", {
      x: 0.55, y: 0.62, w: 12, h: 0.45,
      fontFace: FONT_H, fontSize: 26, bold: true, color: C.ink, margin: 0,
    });
    const b = [
      ["1", "Two clocks", "Same as list A #3, but as a split: move from the middle moment, stop-alarm from the late moment. Not one average wait."],
      ["2", "Read the map, not the wires", "The mini brain already has a grid like a fly’s eye. Where the cube sits on that grid can mean left/right/down. The output wires can stay a diary plus a stop button."],
      ["3", "A tiny graft", "Do not paste the whole fly brain. Add only the missing helper types that plug into “down,” like a short extra hose, not a new ocean."],
      ["4", "Practice big, run small", "Offline, a bigger brain can teach a tiny cheat-sheet. Live, we still run the small slice. Not a fake map on silent wires."],
      ["5", "Place, not volume", "Ask which spots light up, not how loud the whole room is. Loudness wakes the stop alarm. Place can stay quiet."],
    ];
    b.forEach((it, i) => {
      const y = 1.18 + i * 1.1;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x: 0.55, y, w: 12.25, h: 1.0, fill: { color: C.card }, rectRadius: 0.08, shadow: sh(), line: { type: "none" },
      });
      s.addShape(pres.shapes.OVAL, {
        x: 0.75, y: y + 0.22, w: 0.56, h: 0.56, fill: { color: C.teal }, line: { type: "none" },
      });
      s.addText(it[0], {
        x: 0.75, y: y + 0.28, w: 0.56, h: 0.44,
        fontFace: FONT_H, fontSize: 18, bold: true, color: C.white, align: "center", margin: 0,
      });
      s.addText(it[1], {
        x: 1.5, y: y + 0.12, w: 10.9, h: 0.38,
        fontFace: FONT_H, fontSize: 16, bold: true, color: C.ink, margin: 0,
      });
      s.addText(it[2], {
        x: 1.5, y: y + 0.5, w: 10.9, h: 0.4,
        fontFace: FONT_B, fontSize: 14, color: C.muted, margin: 0,
      });
    });
  }

  // ----- 10 Compare -----
  {
    const s = pres.addSlide();
    s.background = { color: C.cream };
    header(s, "BOTH LISTS");
    s.addText("Keep, drop, or merge", {
      x: 0.55, y: 0.65, w: 12, h: 0.45,
      fontFace: FONT_H, fontSize: 28, bold: true, color: C.ink, margin: 0,
    });
    const rows = [
      ["Two clocks", "Keep. Same idea, now as a split, not a shrug."],
      ["Aim the photo / place not volume", "Keep and merge. Grid location is the strong version of “aim better.”"],
      ["Put the whole missing brain back", "Drop for the live robot. Too slow. Maybe only as a teacher offline."],
      ["Turn the stop alarm down", "Drop as the main plan. Safety is not a knob we average."],
      ["Pretend silent wires are joysticks", "Drop. That would be lying about a fly pick."],
    ];
    rows.forEach((r, i) => {
      const y = 1.3 + i * 1.05;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x: 0.55, y, w: 12.25, h: 0.92, fill: { color: C.card }, rectRadius: 0.08, shadow: sh(), line: { type: "none" },
      });
      s.addText(r[0], {
        x: 0.8, y: y + 0.12, w: 4.4, h: 0.68,
        fontFace: FONT_H, fontSize: 16, bold: true, color: C.navy, valign: "middle", margin: 0,
      });
      s.addText(r[1], {
        x: 5.3, y: y + 0.12, w: 7.2, h: 0.68,
        fontFace: FONT_B, fontSize: 16, color: C.ink, valign: "middle", margin: 0,
      });
    });
  }

  // ----- 11 Recommend -----
  {
    const s = pres.addSlide();
    s.background = { color: C.navy };
    s.addText("Recommended approach", {
      x: 0.7, y: 0.45, w: 12, h: 0.5,
      fontFace: FONT_H, fontSize: 16, bold: true, color: C.gold, margin: 0,
    });
    s.addText("Read the map. Use two clocks.", {
      x: 0.7, y: 1.05, w: 12, h: 0.85,
      fontFace: FONT_H, fontSize: 32, bold: true, color: C.white, margin: 0,
    });
    s.addText("First: from the same cube photos we already have, measure where the cube sits on the fly-eye grid. Check if that location matches the human pick (left/right/down). No robot motion for this.\n\nSecond: keep the stop alarm, but read it later than the move. Do not grow the live mini brain to 166,000 cells. Do not invent a fake “down” from silent wires.\n\nIf the grid still cannot match a real pick, we stop calling this a fly pick on the robot — not another millimetre twitch.", {
      x: 0.7, y: 2.1, w: 12, h: 4.4,
      fontFace: FONT_B, fontSize: 18, color: C.sky, margin: 0,
    });
  }

  // ----- 12 Next -----
  {
    const s = pres.addSlide();
    s.background = { color: C.cream };
    header(s, "WHAT YOU DO NOT HAVE TO DO");
    s.addText("Nobody needs to stand by the arm for this", {
      x: 0.55, y: 0.65, w: 12, h: 0.55,
      fontFace: FONT_H, fontSize: 26, bold: true, color: C.ink, margin: 0,
    });
    const nos = [
      [C.coral, "times", "No more tiny robot moves to “see what happens.”"],
      [C.navy, "check", "Yes: replay the old pick videos on a computer."],
      [C.teal, "check", "Yes: test the grid idea on those videos first."],
    ];
    nos.forEach((n, i) => {
      const y = 1.5 + i * 1.55;
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
        x: 0.55, y, w: 12.25, h: 1.4, fill: { color: n[0] }, rectRadius: 0.1, line: { type: "none" },
      });
      s.addShape(pres.shapes.OVAL, {
        x: 0.85, y: y + 0.4, w: 0.6, h: 0.6, fill: { color: C.white }, line: { type: "none" },
      });
      s.addImage({ data: icons[n[1]], x: 0.98, y: y + 0.53, w: 0.34, h: 0.34 });
      s.addText(n[2], {
        x: 1.7, y: y + 0.35, w: 10.7, h: 0.7,
        fontFace: FONT_H, fontSize: 22, bold: true, color: C.white, valign: "middle", margin: 0,
      });
    });
  }

  await pres.writeFile({ fileName: path.join(__dirname, "fly-pick-teen-explainer.pptx") });
  console.log("wrote docs/fly-pick-teen-explainer.pptx");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
