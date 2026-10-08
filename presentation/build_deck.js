// Seminar deck, built in the supervisor's order:
// problem -> data -> representation -> simple setup (phantom) -> one change at a time
// (what / why / result) -> naive vs final -> validation -> state of the art -> conclusion.
// Real data and our own results only; every technical term is explained in the notes.
//
//   cd presentation && node build_deck.js
const path = require("path");
const pptxgen = require("pptxgenjs");

const SKILL_THEME = "C:/Users/HP/AppData/Roaming/Claude/local-agent-mode-sessions/skills-plugin/2f257813-b922-4324-81a1-b7bf94e1d78b/7b622470-5135-45a2-8466-2a58dfb54669/skills/pptx/scripts/apply_theme.js";
const FIG = (f) => path.join(__dirname, "figs", f);
const OUT = path.join(__dirname, "..", "Gaussian_Blobs_Microscopy_v3.pptx");

const THEME = {
  name: "Fluorescence",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "15171C", lt1: "FFFFFF", dk2: "3A3F4B", lt2: "F1F3F6",
    accent1: "E8730C", accent2: "2A78D6", accent3: "1BAF7A",
    accent4: "D64545", accent5: "6B7280", accent6: "F2B134",
    hlink: "2A78D6", folHlink: "6B7280",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.author = "Syed Muhammad Asad";
pres.title = "Gaussian blobs for 3D microscopy";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "DARK",
  background: { color: HEX.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.5, w: 5.4, h: 1.6,
        fontSize: 34, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 3.25, w: 5.4, h: 1.5,
        fontSize: 16, color: "C9CDD6", valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "DARK_LIST",
  background: { color: HEX.dk1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.45, w: 6.0, h: 0.8,
        fontSize: 32, bold: true, color: C.background1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 1.5, w: 6.0, h: 3.5,
        fontSize: 17, color: "E5E7EB", valign: "top", align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.lt1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.28, w: 9.0, h: 0.7,
        fontSize: 28, bold: true, color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "Gaussian blobs for 3D microscopy  ·  MLCV seminar, TU Dresden",
        options: { x: 0.5, y: 5.2, w: 6.5, h: 0.25, fontSize: 9, color: C.accent5, margin: 0 } } },
  ],
  slideNumber: { x: 9.0, y: 5.2, w: 0.5, h: 0.25, fontSize: 9, color: C.accent5, align: "right" },
});

let n = 0;
const addSection = (title) => pres.addSection({ title });
const content = (section, title) => {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  n += 1;
  currentTitle = title;
  return s;
};

// text helpers
const txt = (s, text, o) => s.addText(text, Object.assign({ isTextBox: true, margin: 0, fontSize: 16, color: C.text1, valign: "top" }, o));
const card = (s, x, y, w, h, name) => s.addShape(pres.shapes.ROUNDED_RECTANGLE,
  { x, y, w, h, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: name });

// "what / why / result" column, used on every step slide
function steps(s, rows, x = 6.0, w = 3.5) {
  let y = 1.15;
  const h = 1.25;
  rows.forEach(([label, body], i) => {
    card(s, x, y, w, h, `step-card-${i}`);
    txt(s, label, { x: x + 0.18, y: y + 0.1, w: w - 0.36, h: 0.25, fontSize: 12, bold: true, color: C.accent1 });
    txt(s, body, { x: x + 0.18, y: y + 0.38, w: w - 0.36, h: h - 0.46, fontSize: 14 });
    y += h + 0.07;
  });
}

// speaker notes on the slide, and the same text collected for PRESENTER_NOTES.md
const ALL_NOTES = [];
let currentTitle = "";
const notes = (s, say, terms, asked) => {
  ALL_NOTES.push({ n, title: currentTitle, say, terms, asked: asked || [] });
  s.addNotes(`SAY:\n${say}\n\nTERMS ON THIS SLIDE:\n${terms.map((t) => "- " + t).join("\n")}` +
    (asked && asked.length ? `\n\nIF ASKED:\n${asked.map((q) => "- " + q).join("\n")}` : ""));
};

const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt",
  legendFontFace: "+mn-lt", titleFontFace: "+mn-lt", catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.dk2,
  catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, dataLabelFontSize: 12, dataLabelColor: HEX.dk1,
  valGridLine: { color: "E5E7EB", size: 0.5 }, catGridLine: { style: "none" } };

// ================================================================ 1. title
addSection("Problem and data");
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: "Problem and data" });
  n += 1;
  currentTitle = "Can 3D Gaussian blobs replace microscopy volumes?";
  s.addText(currentTitle, { placeholder: "title" });
  s.addText([
    { text: "Syed Muhammad Asad, Akim", options: { breakLine: true } },
    { text: "MLCV seminar team project, TU Dresden", options: { breakLine: true } },
    { text: "Supervisor: Prof. Martin Weigert" },
  ], { placeholder: "body" });
  s.addImage({ path: FIG("title_tribolium.png"), x: 7.05, y: 0.35, w: 2.55, h: 4.85, objectName: "title-tribolium" });
  txt(s, "Tribolium embryo, light-sheet microscope (real data)", { x: 6.5, y: 5.25, w: 3.2, h: 0.25, fontSize: 10, color: "9CA3AF", align: "right" });
  notes(s,
    "Our question: can we store and show a 3D microscope image as a set of soft 3D blobs instead of millions of voxels, and does the biology, the cell nuclei, survive? I will go from the problem, to the data, to the simplest working setup, then one change at a time, and finally compare against the best existing tool.",
    ["3D Gaussian blob: a soft, fuzzy ball (or stretched ball) of brightness, brightest in the middle and fading outwards. 'Gaussian' only describes how it fades.",
     "Voxel: a 3D pixel, one brightness value at one point of the volume.",
     "Light-sheet microscope: a microscope that lights the sample with a thin sheet of light and images it layer by layer, which gives fast, gentle 3D recordings of living embryos."],
    ["Why Gaussians? They are smooth, each needs only 11 numbers, and they can be drawn very fast on a graphics card."]);
}

// ================================================================ 2. problem
{
  const s = content("Problem and data", "Microscopy volumes are too big to handle easily");
  const stats = [["37 million", "voxels in one Tribolium frame"], ["2.5 GB", "one C. elegans recording (195 frames)"], ["terabytes", "one long light-sheet experiment"]];
  stats.forEach(([big, small], i) => {
    const x = 0.5 + i * 3.05;
    card(s, x, 1.25, 2.85, 1.9, `stat-${i}`);
    txt(s, big, { x: x + 0.2, y: 1.45, w: 2.45, h: 0.8, fontSize: 36, bold: true, color: C.accent1, fontFace: "Cambria" });
    txt(s, small, { x: x + 0.2, y: 2.3, w: 2.45, h: 0.7, fontSize: 15, color: C.text2 });
  });
  txt(s, "Question: can a few thousand 3D blobs store and show such a volume, and still keep every cell nucleus?",
    { x: 0.5, y: 3.6, w: 9.0, h: 0.9, fontSize: 20, bold: true, color: C.text1 });
  notes(s,
    "One frame of our Tribolium data has 37 million voxels. One C. elegans recording from the Cell Tracking Challenge has 195 frames, about 2.5 gigabytes even in 8-bit. Real light-sheet experiments run for hours and produce terabytes. That is hard to store, share and look at. So the question: can a much smaller set of blobs stand in for the volume, for storage and for viewing, without losing the nuclei that biologists actually measure?",
    ["Frame: one 3D snapshot at one time point; a recording is many frames over time.",
     "8-bit: each voxel stored as a number from 0 to 255 (one byte).",
     "Cell nucleus: the compact, DNA-filled centre of a cell. In these images the nuclei are labelled with a fluorescent marker, so they appear as bright blobs."],
    ["Where do the numbers come from? 37 million = 71 x 1024 x 512 voxels (Tribolium frame). 2.5 GB = 195 frames x 12.7 MB (C. elegans, 35 x 512 x 708 voxels, 8-bit)."]);
}

// ================================================================ 3. data
{
  const s = content("Problem and data", "Three real embryo recordings");
  s.addImage({ path: FIG("data_three.png"), x: 0.5, y: 1.1, w: 9.0, h: 3.12, objectName: "data-three" });
  const cap = [["Tribolium", "1 frame, own reference nuclei from a detector"], ["Drosophila", "Cell Tracking Challenge, nuclei marked by hand"], ["C. elegans", "Cell Tracking Challenge, 12 frames, 2,958 nuclei marked by hand"]];
  cap.forEach(([a, b], i) => {
    txt(s, [{ text: a, options: { bold: true, breakLine: true } }, { text: b }],
      { x: 0.5 + i * 3.05, y: 4.3, w: 2.85, h: 0.8, fontSize: 14, color: C.text2 });
  });
  notes(s,
    "Three real light-sheet recordings of developing embryos, shown as maximum projections. Tribolium, a beetle, was our main development data. Drosophila and C. elegans come from the Cell Tracking Challenge, a public benchmark where people have marked the nuclei by hand. Those hand-made marks are what we use to check results honestly. The C. elegans data, 12 frames with almost 3,000 marked nuclei, is what we used for the final comparison against the state of the art.",
    ["Maximum projection: a flat picture of a 3D volume made by keeping, along each line of sight, the brightest voxel. It shows the whole embryo at once.",
     "Cell Tracking Challenge: a public collection of microscopy time-lapse recordings with expert annotations, used to compare methods fairly.",
     "Marked by hand (ground truth): positions of nuclei clicked by human experts. Results checked against these are not circular.",
     "Detector-derived reference: positions found by a program, not a person. Useful during development, but a method can look good just by agreeing with the detector."],
    ["Why do the voxel sizes matter? The depth step is much larger than the pixel size (e.g. 3.0 vs 0.69 micrometres for Tribolium), so nuclei look stretched in depth. Every setting must therefore be given in micrometres, not voxels."]);
}

// ================================================================ 4. representation
addSection("Simple setup and changes");
{
  const s = content("Simple setup and changes", "A volume written as a sum of blobs");
  txt(s, "volume ≈ blob 1 + … + blob K", { x: 0.5, y: 1.2, w: 4.4, h: 0.45, fontSize: 19, bold: true, fontFace: "Cambria" });
  const rows = [["position", "3"], ["size along 3 axes", "3"], ["rotation", "4"], ["brightness", "1"], ["numbers per blob", "11"]];
  s.addTable(rows.map(([a, b], i) => [
    { text: a, options: { bold: i === 4, color: HEX.dk1 } },
    { text: b, options: { bold: true, align: "right", color: i === 4 ? HEX.accent1 : HEX.dk1 } }]),
  { x: 0.5, y: 1.85, w: 4.3, colW: [3.2, 1.1], fontSize: 16, fontFace: "Calibri", rowH: 0.42,
    border: { type: "solid", pt: 0.5, color: "E5E7EB" }, fill: { color: HEX.lt1 } });
  txt(s, "250 blobs = 2,750 numbers instead of 1,048,576 voxels", { x: 0.5, y: 4.15, w: 4.4, h: 0.6, fontSize: 15, color: C.text2 });
  s.addImage({ path: FIG("real_vs_fit.png"), x: 5.05, y: 1.25, w: 4.45, h: 2.25, objectName: "real-vs-fit" });
  txt(s, "Real Tribolium crop and its fit (maximum projections)", { x: 5.05, y: 3.6, w: 4.45, h: 0.3, fontSize: 11, color: C.accent5 });
  notes(s,
    "Every blob is described by 11 numbers: where it is, how wide it is along three axes, how it is rotated, and how bright it is. Adding all blobs together gives back a full volume, which we compare to the real data voxel by voxel. On the right, a real crop and its reconstruction from 250 blobs: the nuclei are there, the fine texture is smoothed away. 250 blobs need 2,750 numbers instead of about a million voxels.",
    ["Rotation (4 numbers): stored as a quaternion, a standard 4-number way to describe a 3D rotation without the jumps that angles can have.",
     "Reconstruction: the volume we get back by drawing all blobs.",
     "K: the number of blobs we allow (the budget)."],
    ["Is 2,750 vs 1,048,576 a file-size ratio? No, it is a count of stored numbers; file sizes depend on how each is stored. The file-size comparison comes on the state-of-the-art slides.",
     "How is the fit done? Start with K blobs, compare their sum with the data, and nudge all 11 x K numbers to reduce the difference (gradient descent with the Adam optimiser), for 3,000 steps."]);
}

// ================================================================ 5. phantom
{
  const s = content("Simple setup and changes", "Step 0 · A phantom with a known answer");
  s.addImage({ path: FIG("phantom.png"), x: 0.5, y: 1.2, w: 5.2, h: 2.6, objectName: "phantom" });
  txt(s, "Synthetic volume, maximum projections", { x: 0.5, y: 3.9, w: 5.2, h: 0.3, fontSize: 11, color: C.accent5 });
  steps(s, [
    ["SETUP", "Fit round blobs to a volume that is itself made of blobs; compare at every voxel"],
    ["WHY", "If the code is right, it must recover a known answer"],
    ["RESULT", "47.9 dB, almost perfect. The usual splatting renderer scored −7.1 dB"],
  ]);
  notes(s,
    "Before touching real data we built the simplest possible test: a synthetic volume made of blobs, so we know the right answer. We fit blobs to it by comparing the blob sum with the volume at every voxel. It recovers the phantom almost perfectly, 47.9 dB. We also learned something here: the standard 3D Gaussian splatting renderer, designed for photographs of solid objects, treats blobs as blocking the ones behind them. Our volumes are transparent and glowing, so that rule is wrong; scored that way the same kind of fit collapsed to minus 7 dB. So from here on we compare blobs to voxels directly.",
    ["Phantom: artificial test data with a known answer.",
     "PSNR (peak signal-to-noise ratio), in dB: a number for how close the reconstruction is to the data; higher is better, +3 dB roughly halves the squared error. It says nothing about whether nuclei survive.",
     "3D Gaussian splatting: a 2023 graphics method that represents scenes as blobs and draws them for a camera.",
     "Alpha-blending (alpha-compositing): the renderer rule that each blob partly hides the blobs behind it. Correct for opaque surfaces, wrong for glowing transparent volumes.",
     "Comparing at every voxel (voxel query): evaluate the blob sum at each voxel centre and compare with the data. No camera, no hiding."],
    ["Why −7.1 dB and 47.9 dB on different phantoms? −7.1 vs 31.7 dB was the controlled comparison on a 15-blob phantom (same fit, two ways of scoring). 47.9 dB is the final phantom fit with 2,000 blobs."]);
}

// ================================================================ 6a. Akim: phantom starting positions
{
  const s = content("Simple setup and changes", "Step 1 · Start blobs on bright peaks");
  s.addChart(pres.charts.BAR, [{ name: "PSNR (dB)",
    labels: ["random start", "random, more steps", "start on peaks", "peaks, more steps"], values: [31.97, 33.65, 53.05, 55.12] }],
  Object.assign({}, chartText, { x: 0.5, y: 1.15, w: 5.2, h: 3.55, barDir: "col", chartColors: [HEX.accent2],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMinVal: 0, valAxisMaxVal: 60,
    valAxisLabelFormatCode: "0", showLegend: false, showTitle: true, catAxisLabelFontSize: 11,
    title: "Akim's phantom experiments: best PSNR (dB) of 8 fits each", titleFontSize: 12, titleColor: HEX.dk2, barGapWidthPct: 50 }));
  steps(s, [
    ["WHAT AKIM CHANGED", "Random starting positions → one blob on each bright peak"],
    ["WHY", "A blob that starts far from any structure wastes the whole fit"],
    ["RESULT", "+21 dB on the phantom (32 → 53); more steps alone gave only +1.7"],
  ]);
  notes(s,
    "This is my teammate Akim's work, from his archive of 57 experiments. On the phantom, he compared random starting positions with starting each blob on a bright peak of the data. More fitting steps barely helped random starts, plus 1.7 dB. Starting on peaks jumped from 32 to 53 dB, and with 3,000 steps to 55. So the very first lesson was: where the blobs start matters more than how long you fit.",
    ["Starting position (initialisation): where each blob is placed before fitting begins.",
     "Bright peak (local maximum): a voxel brighter than all its neighbours; a quick guess for where a blob belongs.",
     "Fitting steps (iterations): how many times all blob numbers are adjusted.",
     "Best of 8 fits: each setting was run with four blob shapes and two budgets (50 and 100 blobs); the bar shows the best one."],
    ["Why does a random start fail? Later we found the reason: blobs barely move during fitting with the default step size (Step 5).",
     "Who did what? Steps 1 and 2 are Akim's experiments; Steps 0 and 3-6 are ours; the comparison with the state of the art is ours."]);
}

// ================================================================ 6b. Akim: real-data sweep
{
  const s = content("Simple setup and changes", "Step 2 · Real data: stretched, rotated blobs win");
  s.addChart(pres.charts.BAR, [{ name: "median PSNR (dB)",
    labels: ["fixed round", "learned round", "stretched", "stretched + rotated"], values: [6.82, 20.36, 23.32, 23.96] }],
  Object.assign({}, chartText, { x: 0.5, y: 1.15, w: 5.2, h: 3.55, barDir: "col", chartColors: [HEX.accent2],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMinVal: 0, valAxisMaxVal: 30,
    valAxisLabelFormatCode: "0", showLegend: false, showTitle: true, catAxisLabelFontSize: 11,
    title: "Akim's real Tribolium sweep: median PSNR (dB), ~75 fits per shape", titleFontSize: 12, titleColor: HEX.dk2, barGapWidthPct: 50 }));
  steps(s, [
    ["WHAT AKIM TESTED", "43 experiments, 299 fits: 4 blob shapes, 50-1,000 blobs, sampling, loss, automatic settings"],
    ["WHY", "Real nuclei are not the phantom: stretched, unevenly bright, packed tightly"],
    ["RESULT", "Stretched + rotated blobs best; top fit 25.9 dB with 750-1,000 blobs on the whole volume"],
  ]);
  notes(s,
    "Then Akim moved to the real Tribolium data: 43 experiments and 299 fits. He compared four blob shapes: a fixed round blob, a round blob whose size is learned, a stretched blob, and a stretched blob that can also rotate. Fixed round blobs basically fail, 6.8 dB. Stretched and rotated blobs are best, median 24 dB. He also varied the number of blobs from 50 to 1,000, how voxels are sampled during fitting, the loss, and an automatic tuner that reads settings from the data. His best fit was 25.9 dB with 750 to 1,000 blobs on the whole volume. He also tried splitting blobs during fitting, densification, which did not beat a fixed budget, and matching the number of blobs to the number of cells, which found cells with F1 0.76.",
    ["Fixed round blob: every blob has the same, unchangeable size; only position and brightness are fitted.",
     "Learned round blob: one size per blob, fitted.",
     "Sampling: each fitting step looks at a random subset of voxels (here 50,000), partly chosen from bright regions.",
     "Loss: the number the fit tries to reduce, here the squared difference between blobs and data (plus variants Akim tested).",
     "Automatic settings (auto-tuner): Akim's tool that estimates background, noise, nucleus size and spacing from the data.",
     "Densification: splitting or cloning blobs during fitting where the error is large (a standard 3D-splatting trick)."],
    ["Why is fixed round so bad? Its size was fixed at 2 voxels while real nuclei need 5-7, so it cannot cover them; part of the gap is that constant, not roundness itself.",
     "Why median, not best? One run per setting can be lucky; the median over ~75 fits is more robust. Our next step re-tested this with repeated seeds."]);
}

// ================================================================ 6. step 3 shape re-test
{
  const s = content("Simple setup and changes", "Step 3 · Repeats confirm: stretching is real");
  s.addChart(pres.charts.BAR, [{ name: "PSNR (dB)", labels: ["round", "stretched", "stretched + rotated"], values: [22.31, 25.21, 25.51] }],
    Object.assign({}, chartText, { x: 0.5, y: 1.15, w: 5.2, h: 3.55, barDir: "col", chartColors: [HEX.accent2],
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisMinVal: 20, valAxisMaxVal: 26,
      valAxisLabelFormatCode: "0", showLegend: false, showTitle: true, title: "Real Tribolium crops: PSNR (dB), axis from 20",
      titleFontSize: 12, titleColor: HEX.dk2, barGapWidthPct: 60 }));
  steps(s, [
    ["WHAT WE DID", "Re-ran round, stretched and stretched + rotated blobs with 5 seeds and 3 budgets each"],
    ["WHY", "Akim had one run per setting; a single run can be luck"],
    ["RESULT", "+3.2 dB with small spread; learned blobs are as stretched as real nuclei"],
  ]);
  notes(s,
    "Our first own step was to check Akim's shape result with repeats: the same three blob shapes, each fitted 15 times, with 5 random seeds and 3 blob budgets, on the same kind of real crops. The result holds: 22.3 dB for round blobs, 25.5 for stretched and rotated, and the spread between repeats is small. Most of the gain comes from stretching; rotation adds half a dB. And the blobs did what we expected: their depth-to-width ratio, 0.79, is close to the ratio measured on the real nuclei, 0.73. Real nuclei look stretched in depth partly because the microscope blurs more along that axis.",
    ["Round (isotropic) blob: same size in every direction.",
     "Stretched (anisotropic, axis-aligned) blob: three independent sizes along x, y and z.",
     "Rotation: lets the stretched blob point in any direction.",
     "Seed: the random number that decides the starting point of a fit; different seeds test whether a result is luck."],
    ["Did this also help finding nuclei? Slightly: nucleus F1 0.745 (round) vs 0.805 (stretched + rotated), significant in a paired test (p = 0.006, 5 seeds).",
     "Why does the axis start at 20? To make a 3 dB difference visible; it is stated on the chart."]);
}

// ================================================================ 7. step 2 seeding
{
  const s = content("Simple setup and changes", "Step 4 · Start each blob on a nucleus");
  s.addChart(pres.charts.BAR, [
    { name: "Brightest-peak start", labels: ["blob at a nucleus at start", "nuclei found after fitting"], values: [73, 52] },
    { name: "One start per nucleus", labels: ["blob at a nucleus at start", "nuclei found after fitting"], values: [100, 71] }],
  Object.assign({}, chartText, { x: 0.5, y: 1.15, w: 5.2, h: 3.55, barDir: "col", chartColors: [HEX.accent5, HEX.accent2],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0\"%\"", valAxisMinVal: 0, valAxisMaxVal: 110,
    valAxisLabelFormatCode: "0\"%\"", showLegend: true, legendPos: "b", legendFontSize: 12, showTitle: true,
    title: "4 Tribolium regions, 110 nuclei", titleFontSize: 12, titleColor: HEX.dk2, barGapWidthPct: 50 }));
  steps(s, [
    ["WHAT WE CHANGED", "Bright-peak starts → one starting blob on every detected nucleus first"],
    ["WHY", "During fitting blobs moved less than 1 voxel; nuclei are ~20 voxels apart"],
    ["RESULT", "Every nucleus gets a blob (73% → 100%); found nuclei 52% → 71%"],
  ]);
  notes(s,
    "Building on Akim's peak starts. We measured how far blobs travel during fitting: less than one voxel on average, while nuclei are about 20 voxels apart. So a blob that starts between nuclei stays there. Akim's start put blobs on the brightest peaks, and bright background crowded out faint nuclei: only 73% of nuclei had a blob at the start. Starting one blob per detected nucleus covers all of them and raises the share of nuclei found after fitting from 52 to 71%.",
    ["Detected nucleus / peak: a local brightness maximum found by a simple detector before fitting.",
     "Peak-search window (non-maximum suppression): around each peak, weaker peaks closer than a set distance are ignored, so one nucleus gives one peak.",
     "Found after fitting (recall): the share of reference nuclei that a detector still finds in the reconstruction."],
    ["Is this circular? Partly, on Tribolium: the reference nuclei came from a related detector. On hand-marked Drosophila nuclei the 'window = nucleus spacing' rule did NOT transfer; what transferred was setting windows in micrometres, not voxels.",
     "Why did blobs not move? See the next step: the step size was too small."]);
}

// ================================================================ 8. step 3 learning rate
{
  const s = content("Simple setup and changes", "Step 5 · Bigger steps let blobs travel");
  s.addChart(pres.charts.LINE, [{ name: "blobs that moved to another nucleus", labels: ["1×", "3×", "10×", "31×", "62×"], values: [1.0, 5.5, 14.5, 28.0, 32.0] }],
    Object.assign({}, chartText, { x: 0.5, y: 1.15, w: 5.2, h: 3.55, chartColors: [HEX.accent2], lineSize: 3, lineDataSymbolSize: 9,
      showValue: true, dataLabelPosition: "t", dataLabelFormatCode: "0\"%\"", valAxisMinVal: 0, valAxisMaxVal: 40,
      valAxisLabelFormatCode: "0\"%\"", showLegend: false, showTitle: true,
      title: "Blobs that moved to another nucleus vs position step size (×default)", titleFontSize: 12, titleColor: HEX.dk2,
      showCatAxisTitle: true, catAxisTitle: "position step size, multiple of the default", catAxisTitleFontSize: 11, catAxisTitleColor: HEX.dk2 }));
  steps(s, [
    ["WHAT WE CHANGED", "Raised the step size for blob positions up to 62×"],
    ["WHY", "The default was copied from photo-scene code and never scaled to voxel units"],
    ["RESULT", "Badly started fits catch up (F1 0.64 → 0.78), but good starts get worse (0.84 → 0.78)"],
  ]);
  notes(s,
    "Why did blobs not move? The step size for positions was taken from 3D-splatting code made for photo scenes, where coordinates are small, and we never rescaled it for voxel coordinates. Raising it lets blobs travel: one in three moves to a different nucleus. With bad starting points this closes the gap completely. But with good starting points it slightly hurts. So the lesson is a trade-off: good starts with small steps is still best, at F1 0.84.",
    ["Learning rate / step size: how far each fitting step moves the numbers. Too small: blobs barely move. Too large: they overshoot.",
     "F1: one number combining 'how many real nuclei were found' and 'how many detections were real'; 1.0 is perfect.",
     "Moved to another nucleus (migration): the nearest nucleus of a blob changed between start and end."],
    ["Does that undo Steps 1 and 4? It reframes them: starting points mattered so much because blobs could not move. With a suitable step size, a random start reaches the same F1."]);
}

// ================================================================ 9. step 4 size cap
{
  const s = content("Simple setup and changes", "Step 6 · Faint nuclei: cap the blob size");
  s.addImage({ path: FIG("size_cap.png"), x: 0.5, y: 1.2, w: 5.2, h: 2.6, objectName: "size-cap" });
  txt(s, "Green: nucleus found  ·  red: missed", { x: 0.5, y: 3.9, w: 5.2, h: 0.3, fontSize: 11, color: C.accent5 });
  steps(s, [
    ["WHAT WE CHANGED", "Limited how wide a nucleus blob may grow (10 voxels)"],
    ["WHY", "Lost faint nuclei had blobs ~30% wider, spread out until no peak was left"],
    ["RESULT", "Faint nuclei found: 13 → 19 of 31. Image quality unchanged; 1 of 4 regions worse"],
  ]);
  notes(s,
    "We then asked which nuclei are still lost. They are the faint ones, and when we followed their blobs during fitting, the blobs did not fade; they spread out, about 30 percent wider than for nuclei we kept, until there was no peak left to detect. So we capped the width of nucleus blobs. Over four regions, faint nuclei found went from 13 of 31 to 19 of 31, and image quality changed by only 0.01 dB. Honestly: one region got worse, and on the Drosophila data the cap changed nothing.",
    ["Faint nucleus: one whose brightness above the surroundings is in the lower half.",
     "Size cap: an upper limit on a blob's size, applied only to the blobs that started on nuclei.",
     "Region: one 64 x 128 x 128 crop of the Tribolium frame."],
    ["Is +6 nuclei significant? It is one seed and 4 regions, so treat it as promising, not settled. The result survived a scoring-bug fix (it was +7 before)."]);
}

// ================================================================ 10. naive vs final
addSection("Results and validation");
{
  const s = content("Results and validation", "From the simple setup to the final recipe");
  const head = ["Change", "By", "Measured on", "Before", "After"].map((t) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 } } }));
  const body = [
    ["Start on bright peaks", "Akim", "PSNR, phantom", "32.0 dB", "53.1 dB"],
    ["Stretched + rotated blobs", "Akim", "PSNR, real data, median", "20.4 dB", "24.0 dB"],
    ["Repeats of the shape test", "us", "PSNR, 15 fits", "22.3 dB", "25.5 dB"],
    ["One start per nucleus", "us", "nuclei found, of 110", "52%", "71%"],
    ["Bigger position steps", "us", "F1, random starts", "0.64", "0.78"],
    ["Size cap", "us", "faint nuclei found, of 31", "13", "19"],
  ].map((r) => r.map((t, j) => ({ text: t, options: { bold: j === 4, color: j === 4 ? HEX.accent1 : HEX.dk1, align: j >= 3 ? "center" : "left" } })));
  s.addTable([head].concat(body), { x: 0.5, y: 1.15, w: 9.0, colW: [2.75, 0.75, 2.6, 1.45, 1.45], fontSize: 15, fontFace: "Calibri",
    rowH: 0.43, border: { type: "solid", pt: 0.5, color: "E5E7EB" }, fill: { color: HEX.lt1 } });
  txt(s, "Each change measured on its own, against the version before it; the numbers are not added together.",
    { x: 0.5, y: 4.35, w: 9.0, h: 0.45, fontSize: 15, color: C.text2 });
  notes(s,
    "Here are all six changes side by side, two by Akim and four by us, each compared with the version just before it. I want to be clear that these were measured one at a time, on different data and with different numbers of repeats, so you cannot add them up into one overall improvement. Together they turn a fit that merely looks right into one where the nuclei are found much more reliably.",
    ["Naive / simple setup: round blobs with random starting positions, default step size, no cap."],
    ["Why not one combined number? Each experiment used a different number of regions, seeds and references; combining them would hide that."]);
}

// ================================================================ 11. validation
{
  const s = content("Results and validation", "Does the fit behave as expected?");
  const cards = [
    ["0.79 vs 0.73", "depth-to-width ratio: learned blobs vs measured nuclei"],
    ["14 vs 10", "hand-marked Drosophila nuclei in the top 100 detections: fitted blobs vs raw image"],
    ["≤ 4%", "difference in nuclei found between three repeated fits"],
  ];
  cards.forEach(([big, small], i) => {
    const x = 0.5 + i * 3.05;
    card(s, x, 1.25, 2.85, 2.6, `check-${i}`);
    txt(s, big, { x: x + 0.2, y: 1.45, w: 2.45, h: 0.8, fontSize: 32, bold: true, color: C.accent3, fontFace: "Cambria" });
    txt(s, small, { x: x + 0.2, y: 2.35, w: 2.45, h: 1.35, fontSize: 15, color: C.text2 });
  });
  txt(s, "Limit: with all 457 of its candidates the raw-image detector finds 26 of 29, more than any fit.",
    { x: 0.5, y: 4.2, w: 9.0, h: 0.6, fontSize: 15, color: C.text2 });
  notes(s,
    "Three checks that the fit does what we think. First, shape: the learned blobs are about as stretched as the real nuclei. Second, the honest test against people's annotations: on Drosophila, among the 100 most confident detections, the fitted blobs contain 14 of 29 hand-marked nuclei, the raw image only 10. So fitting ranks real nuclei higher. Third, fits are repeatable. And the limit: if the plain detector may use all its candidates it finds 26 of 29, more than any fit. So this is better ranking, not a better detector.",
    ["Top 100 detections: compare methods at the same number of guesses, because only some nuclei are marked and extra detections cannot be judged as wrong.",
     "Raw-image detector: the same peak detector run directly on the original data.",
     "Repeated fits: the same data fitted three times; for the Luxar fits used later, nuclei kept differed by at most 2.1-3.7 points depending on the detector."],
    ["How noisy is 14 vs 10? One nucleus is 3.4% of 29; we only call a difference real if it is about 3 nuclei and has the same sign in all seeds (mean of 3 seeds here)."]);
}

// ================================================================ 12. SOTA setup
addSection("State of the art");
{
  const s = content("State of the art", "Against the state of the art, at the same file size");
  const L = require(FIG("four_labels.json"));
  const four = [["raw", "Original"], ["jpegxl", `JPEG-XL ${L.jpegxl[0]}×`], ["jpeg2k", `JPEG2000 ${L.jpeg2k[0]}×`], ["luxar", `Luxar ${L.luxar[0]}×`]];
  four.forEach(([f, label], i) => {
    const x = 0.5 + i * 2.27;
    txt(s, label, { x, y: 1.12, w: 2.2, h: 0.3, fontSize: 14, bold: true, align: "center" });
    s.addImage({ path: FIG(`four_${f}.png`), x, y: 1.45, w: 2.2, h: 1.68, objectName: `four-${f}` });
    if (f !== "raw") txt(s, `${L[f][1]}% of nuclei kept`, { x, y: 3.15, w: 2.2, h: 0.28, fontSize: 12, color: C.accent5, align: "center" });
  });
  const pts = [["Luxar (2026)", "state-of-the-art Gaussian-blob tool for microscopy"], ["JPEG2000", "standard image compression, set to exactly the same file size"], ["Test", "12 C. elegans frames, 2,958 hand-marked nuclei, 4 nucleus detectors"]];
  pts.forEach(([a, b], i) => {
    txt(s, [{ text: a, options: { bold: true, color: HEX.accent1, breakLine: true } }, { text: b }],
      { x: 0.5 + i * 3.05, y: 3.6, w: 2.85, h: 1.3, fontSize: 15, color: C.text1 });
  });
  notes(s,
    "Our own fits were better than our naive start, but are they better than what exists? The strongest existing tool is Luxar, from the Royer lab, which fits Gaussian blobs to large microscopy data. We compared Luxar with ordinary image compression, JPEG2000, squeezed to exactly the same number of bytes, on 12 C. elegans frames with almost 3,000 hand-marked nuclei, and asked four different nucleus detectors how many nuclei survive. The pictures show the same region at about 89 times smaller: the codecs keep the texture, Luxar draws smooth blobs.",
    ["Compression ratio: original size divided by compressed size; 100x means 100 times smaller.",
     "Same file size (matched bytes): every Luxar file is compared with a JPEG2000 file of exactly the same size on disk.",
     "JPEG2000 / JPEG-XL: standard image compression formats; here each 2D slice of the volume is compressed.",
     "Nuclei kept: nuclei found after compression divided by nuclei found in the original by the same detector (1.0 = nothing lost)."],
    ["Why not compare with our own fits? Luxar is faster and better engineered; if blobs are to win anywhere, the strongest blob tool must win.",
     "We found a bug in our own first JPEG2000 setup (it compressed the volume along the wrong axes, about 10 dB worse). Fixing it reversed our first conclusion; a test now prevents it."]);
}

// ================================================================ 13. SOTA result
{
  const s = content("State of the art", "Same file size: the winner depends on the detector");
  s.addChart(pres.charts.BAR, [
    { name: "below 300× smaller", labels: ["LoG", "watershed", "Cellpose 2D", "Cellpose 3D"], values: [1.3, -4.0, -9.4, 3.5] },
    { name: "300× and beyond", labels: ["LoG", "watershed", "Cellpose 2D", "Cellpose 3D"], values: [-16.6, -19.3, -23.2, -18.2] }],
  Object.assign({}, chartText, { x: 0.5, y: 1.15, w: 5.6, h: 3.75, barDir: "col", chartColors: [HEX.accent2, HEX.accent4],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "+0.0;-0.0", valAxisMinVal: -30, valAxisMaxVal: 10,
    valAxisLabelFormatCode: "+0;-0;0", showLegend: true, legendPos: "b", legendFontSize: 12, showTitle: true,
    title: "Luxar minus JPEG2000: nuclei kept (points), mean of 12 frames", titleFontSize: 12, titleColor: HEX.dk2, barGapWidthPct: 40 }));
  card(s, 6.35, 1.25, 3.15, 1.55, "psnr-card");
  txt(s, [{ text: "PSNR: JPEG2000 better", options: { bold: true, breakLine: true } }, { text: "on 12 of 12 frames" }],
    { x: 6.55, y: 1.42, w: 2.8, h: 1.2, fontSize: 18, color: C.text1 });
  card(s, 6.35, 2.95, 3.15, 1.95, "detectors-card");
  txt(s, [{ text: "Detectors disagree", options: { bold: true, breakLine: true } },
    { text: "Cellpose 3D prefers blobs; watershed and Cellpose 2D prefer JPEG2000; all prefer JPEG2000 past 300×" }],
  { x: 6.55, y: 3.1, w: 2.8, h: 1.7, fontSize: 15, color: C.text1 });
  notes(s,
    "The result. Bars above zero mean the blobs kept more nuclei than JPEG2000 at the same file size; below zero, JPEG2000 kept more. Up to about 300 times compression the answer depends on the detector: Cellpose in 3D mode finds 3.5 points more nuclei in the blob version, the classical watershed and Cellpose in 2D mode find more in JPEG2000, and the simple blob detector LoG shows no consistent difference. Beyond 300 times, every detector prefers JPEG2000 by 17 to 23 points. Meanwhile PSNR says JPEG2000 is better on every single frame, so PSNR alone could not have told you which format a given analysis needs. The same Cellpose network changing sides between 2D and 3D mode shows the problem comes from processing slice by slice.",
    ["LoG (Laplacian of Gaussian): a classic blob detector that responds to bright round spots of a given size.",
     "Watershed: a classical segmentation that separates touching bright regions, like water filling valleys.",
     "Cellpose: a widely used deep-learning model that outlines cells or nuclei. 2D mode works slice by slice and joins slices; 3D mode looks at the volume in three directions at once.",
     "Points: percentage points of nuclei kept, e.g. 0.90 vs 0.86 is 4 points.",
     "Statistics: each frame counts once; Wilcoxon signed-rank tests over the 12 frames, Holm-corrected for 8 tests. Everything except LoG below 300x is significant (p ≤ 0.01)."],
    ["Is LoG +1.3 real? No: 8 of 12 frames, p = 0.11; it comes from one embryo only.",
     "Why does JPEG2000 win at high compression? Below about 3 blobs per nucleus, blob fits lose whole nuclei; wavelet compression degrades gradually."]);
}

// ================================================================ 14. cliff
{
  const s = content("State of the art", "Below ~3 blobs per nucleus, nuclei vanish");
  s.addImage({ path: FIG("cliff_deck.png"), x: 0.5, y: 1.2, w: 5.75, h: 3.4, objectName: "cliff" });
  steps(s, [
    ["WHAT WE SEE", "Nuclei kept drops below 90% at 1.1-3.1 blobs per nucleus, on all 12 frames"],
    ["PSNR", "Barely moves over these budgets: at most 2.6 dB per frame"],
    ["SO", "Image quality cannot tell you when the blob budget is too small"],
  ], 6.45, 3.05);
  notes(s,
    "Why do blobs collapse at high compression? Each point is one Luxar fit; the x axis is blobs per hand-marked nucleus. Above about 3 blobs per nucleus almost every nucleus survives; below that, they disappear quickly, on every frame of both embryos. At the same time, PSNR changes very little across these budgets. So if you choose the budget by image quality, you can fall off this cliff without noticing.",
    ["Blobs per nucleus: number of blobs in the fit divided by the number of hand-marked nuclei in that frame.",
     "Cliff: a sudden drop instead of a gradual decline.",
     "90% line: the threshold we fixed in advance to define the cliff (we wrote down '≤ 3.5 blobs per nucleus' before running the extra frames)."],
    ["Why fix thresholds in advance? So we cannot tune the claim to the data afterwards. Two of our advance predictions failed, and we report them as failed."]);
}

// ================================================================ 15. conclusion
addSection("Conclusion");
{
  const s = pres.addSlide({ masterName: "DARK_LIST", sectionTitle: "Conclusion" });
  n += 1;
  currentTitle = "What we learned";
  s.addText(currentTitle, { placeholder: "title" });
  s.addText([
    { text: "Shape, starting points, step size and a size cap each made the fit find more nuclei.", options: { bullet: true, breakLine: true } },
    { text: "Blobs are good for looking, not for measuring: at the same file size a standard codec keeps nuclei as well or better.", options: { bullet: true, breakLine: true } },
    { text: "PSNR cannot choose the format: test with the analysis you will actually run.", options: { bullet: true } },
  ], { placeholder: "body", paraSpaceAfter: 14 });
  s.addImage({ path: FIG("title_tribolium.png"), x: 7.05, y: 0.35, w: 2.55, h: 4.85, objectName: "end-tribolium" });
  notes(s,
    "Three take-aways. One: our step-by-step changes made blob fits find nuclei much more reliably. Two: compared with the state of the art at equal file size, blobs are good for viewing but not the right format for measuring; a standard codec keeps the nuclei as well or better. Three: the usual image-quality number cannot decide this; you have to test with the analysis you plan to run.",
    ["Codec: a compression method (coder-decoder)."],
    ["So was the project a failure? No: we now know where blobs should and should not be trusted, which is what a user needs; and the evaluation method itself is reusable."]);
}

// ================================================================ 16. next
{
  const s = content("Conclusion", "Next: compression that biologists can use");
  const items = [
    ["Lossless and near-lossless", "what labs actually store: exact, or errors kept below the camera noise"],
    ["Video codecs", "use the similarity between neighbouring slices and time points"],
    ["Judge by tracking", "measure with the Cell Tracking Challenge scores, not image quality"],
  ];
  items.forEach(([a, b], i) => {
    const y = 1.25 + i * 1.2;
    s.addShape(pres.shapes.OVAL, { x: 0.5, y: y + 0.05, w: 0.6, h: 0.6, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: `num-${i}` });
    txt(s, String(i + 1), { x: 0.5, y: y + 0.05, w: 0.6, h: 0.6, fontSize: 20, bold: true, color: C.background1, align: "center", valign: "middle" });
    txt(s, [{ text: a, options: { bold: true, breakLine: true } }, { text: b, options: { color: HEX.dk2 } }],
      { x: 1.35, y, w: 8.1, h: 1.0, fontSize: 17 });
  });
  notes(s,
    "Following our supervisor's advice, the next step is compression biologists would actually use: lossless storage, or near-lossless where every error stays below the camera noise; video codecs, which exploit the similarity between neighbouring slices and time points; and judging success by whether cells can still be tracked, using the Cell Tracking Challenge's official scores.",
    ["Lossless: the decompressed data is bit-for-bit identical to the original.",
     "Near-lossless / noise-bounded: every voxel may change, but by less than the camera's own noise.",
     "Video codec: compression made for movies (e.g. H.265, AV1) that predicts each frame from its neighbours.",
     "Tracking: following each cell over time, including divisions."],
    ["How much do lossless methods give? Typically 2-4x on microscopy (we measured 4.2x with lossless JPEG-XL on C. elegans); noise-bounded methods reach far more on raw camera data (up to ~100x reported for B3D)."]);
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  try {
    const { applyTheme } = require(SKILL_THEME);
    await applyTheme(OUT, THEME);
  } catch (e) {
    console.warn("theme not applied:", e.message);
  }
  console.log(`wrote ${OUT} (${n} slides)`);
  // presenter notes + glossary as one printable markdown file
  const fs = require("fs");
  const md = ["# Presenter notes: Gaussian blobs for 3D microscopy", "",
    "Generated by presentation/build_deck.js from the same text as the PowerPoint speaker notes.", ""];
  const glossary = new Map();
  ALL_NOTES.forEach(({ n: k, title, say, terms, asked }) => {
    md.push(`## Slide ${k}: ${title}`, "", "**Say:** " + say, "");
    if (terms.length) { md.push("**Terms on this slide:**", ""); terms.forEach((t) => { md.push("- " + t); const [w] = t.split(":"); if (!glossary.has(w)) glossary.set(w, t); }); md.push(""); }
    if (asked.length) { md.push("**If asked:**", ""); asked.forEach((q) => md.push("- " + q)); md.push(""); }
  });
  md.push("## Glossary (all terms, in order of first use)", "");
  glossary.forEach((t) => md.push("- " + t));
  const NL = String.fromCharCode(10);
  fs.writeFileSync(path.join(__dirname, "PRESENTER_NOTES.md"), md.join(NL) + NL, "utf8");
  console.log(`wrote PRESENTER_NOTES.md (${ALL_NOTES.length} slides, ${glossary.size} terms)`);
})();
