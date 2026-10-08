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
  catAxisLabelFontSize: 13, valAxisLabelFontSize: 12, dataLabelFontSize: 13, dataLabelColor: HEX.dk1,
  valGridLine: { color: "E5E7EB", size: 0.5 }, catGridLine: { style: "none" } };

// ================================================================ 1. title
addSection("Problem and data");
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: "Problem and data" });
  n += 1;
  currentTitle = "Can 3D Gaussian blobs replace microscopy volumes?";
  s.addText(currentTitle, { placeholder: "title" });
  s.addText([
    { text: "Syed Muhammad Asad, Akim Al-Makhdar", options: { breakLine: true } },
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
    ["RESULT", "Same fit, 15-blob phantom: 31.7 dB at voxels, −7.1 dB via the splatting renderer. Final fit: 47.9 dB"],
  ]);
  notes(s,
    "Before touching real data we built the simplest possible test: a synthetic volume made of blobs, so we know the right answer. We fit blobs to it by comparing the blob sum with the volume at every voxel. We also learned something here: the standard 3D Gaussian splatting renderer, designed for photographs of solid objects, treats blobs as blocking the ones behind them. Our volumes are transparent and glowing, so that rule is wrong. In a controlled test on a small 15-blob phantom, the same fit scored 31.7 dB when compared at the voxels and minus 7.1 dB through the splatting renderer. So from here on we compare blobs to voxels directly. Our final phantom fit, with 2,000 blobs on a 50-blob phantom, reached 47.9 dB, almost perfect; that is a separate, later run.",
    ["Phantom: artificial test data with a known answer.",
     "PSNR (peak signal-to-noise ratio), in dB: a number for how close the reconstruction is to the data; higher is better, +3 dB roughly halves the squared error. It says nothing about whether nuclei survive.",
     "3D Gaussian splatting: a 2023 graphics method that represents scenes as blobs and draws them for a camera.",
     "Alpha-blending (alpha-compositing): the renderer rule that each blob partly hides the blobs behind it. Correct for opaque surfaces, wrong for glowing transparent volumes.",
     "Comparing at every voxel (voxel query): evaluate the blob sum at each voxel centre and compare with the data. No camera, no hiding."],
    ["Why two phantoms? 31.7 vs −7.1 dB is the controlled comparison (same fit, two ways of scoring, 15-blob phantom). 47.9 dB is a later run (2,000 blobs fitted to a 50-blob phantom) and is not part of that comparison."]);
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
    ["Why does a random start fail? Later we found the reason: blobs barely move during fitting with the default step size (Step 7).",
     "Who did what? Steps 1-4 are Akim's experiments; Steps 0 and 5-8 are ours; the comparison with the state of the art is ours."]);
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

// ================================================================ 6c. Akim: densification
{
  const s = content("Simple setup and changes", "Step 3 · Grow blobs where the error is large");
  s.addImage({ path: FIG("akim_densify.png"), x: 0.5, y: 1.2, w: 5.3, h: 3.25, objectName: "akim-densify" });
  txt(s, "Akim's runs, same Tribolium crop, stretched + rotated blobs", { x: 0.5, y: 4.55, w: 5.3, h: 0.3, fontSize: 11, color: C.accent5 });
  steps(s, [
    ["WHAT AKIM CHANGED", "Start with 25 blobs; add new ones where the error stays high (12 variants)"],
    ["WHY", "Let the data decide how many blobs it needs, instead of guessing"],
    ["RESULT", "Slightly better per blob (34 grown: 24.3 dB vs 50 fixed: 23.8 dB); best 24.9 vs 24.5 dB"],
  ]);
  notes(s,
    "Akim also tried letting the number of blobs grow during fitting. He starts with 25 blobs and, where the error stays high, splits or copies blobs, a standard trick from 3D Gaussian splatting called densification. He tried 12 variants: when to add blobs, whether new blobs inherit their parent's settings, how many voxels to sample. On the same crop, grown fits reached slightly better quality for the same number of blobs, for example 24.3 dB with 34 grown blobs against 23.8 dB with 50 fixed blobs. But the best values are close, 24.9 against 24.5 dB, so it is a modest gain, not a breakthrough, and we kept a fixed budget afterwards because it is simpler to control.",
    ["Densification: adding blobs during fitting by splitting or cloning existing ones where the reconstruction error is large.",
     "Plateau trigger: only add blobs once the error stops improving.",
     "Inherit: a new blob starts with its parent's size and brightness instead of default values.",
     "Fixed budget: the number of blobs is set before fitting and never changes."],
    ["Why not use densification then? The gain was within about half a dB, the final number of blobs is harder to predict, and fixed budgets made the later comparisons cleaner.",
     "Does more fitting time explain it? Partly: the best grown fit used 5,000 steps; the fixed budgets used 3,000."]);
}

// ================================================================ 6d. Akim: cell-count budget
{
  const s = content("Simple setup and changes", "Step 4 · Set the budget from the cell count");
  s.addImage({ path: FIG("akim_cellbudget.png"), x: 0.5, y: 1.2, w: 5.3, h: 3.12, objectName: "akim-cellbudget" });
  txt(s, "Akim's runs, whole Tribolium volume, 230 cells found by a detector", { x: 0.5, y: 4.4, w: 5.3, h: 0.3, fontSize: 11, color: C.accent5 });
  steps(s, [
    ["WHAT AKIM CHANGED", "Count cells first (230), then use ~1.15 blobs per cell + 12% for background (301), and multiples"],
    ["WHY", "If each cell gets a blob, the blobs should match the cells"],
    ["RESULT", "Cell matching F1 0.35 → 0.76 with more blobs, while PSNR falls past ~900 blobs"],
  ]);
  notes(s,
    "Akim's last idea: instead of guessing the number of blobs, count the cells first. A detector found about 230 cells in the whole volume; he then used 1.15 blobs per cell plus 12 percent for the background, which gives 301 blobs, and multiples of that. He then checked how well peaks in the blob reconstruction match the detected cells. Matching improves steadily with more blobs, from F1 0.35 at 301 blobs to 0.76 at 1,804. But look at the right panel: image quality peaks around 900 blobs and then drops. The reason is that all runs stopped after 3,000 steps, which is not enough for a large number of blobs. This was the first sign that image quality and finding cells do not always agree.",
    ["Cell count: the number of cells, here estimated by a detector from the data, not counted by a person.",
     "Cell matching F1: how well peaks in the reconstruction match the detected cells; 1.0 is perfect.",
     "Support blobs (background): the extra blobs not tied to a cell, for the dimmer tissue between nuclei.",
     "Under-trained: the fit stopped before it had finished improving."],
    ["Are the 230 cells ground truth? No, they come from a detector with fixed settings; the hand-marked checks come later (validation slide and the state-of-the-art test).",
     "Does this show that the cell-count rule is a good way to pick the budget? No. F1 rose because the budget grew sixfold, from 301 to 1,804 blobs. The rule was never compared with another way of choosing the budget.",
     "Why does PSNR fall with more blobs? The number of fitting steps was fixed at 3,000; more blobs need more steps. We later saw the same confound in our own experiments."]);
}

// ================================================================ 6. step 5 shape re-test
{
  const s = content("Simple setup and changes", "Step 5 · Repeats confirm: stretching is real");
  // runs/param_inversion_3d/param_inversion.csv: mean of 5 seeds per shape and budget
  const shapes = ["round", "stretched", "stretched + rotated"];
  s.addChart(pres.charts.BAR, [
    { name: "50 blobs", labels: shapes, values: [21.60, 25.07, 25.46] },
    { name: "100 blobs", labels: shapes, values: [22.20, 25.21, 25.54] },
    { name: "250 blobs", labels: shapes, values: [23.14, 25.34, 25.52] }],
    Object.assign({}, chartText, { x: 0.5, y: 1.15, w: 5.2, h: 3.55, barDir: "col",
      chartColors: ["9CC3EE", HEX.accent2, "1D4F8F"], showValue: false, valAxisMinVal: 20, valAxisMaxVal: 26,
      valAxisLabelFormatCode: "0", showLegend: true, legendPos: "b", legendFontSize: 12, showTitle: true,
      title: "Tribolium crops: PSNR (dB), 5 seeds each, axis from 20",
      titleFontSize: 12, titleColor: HEX.dk2, barGapWidthPct: 60 }));
  txt(s, "Seed-to-seed spread (SD) within each bar: 0.02–0.13 dB", { x: 0.5, y: 4.75, w: 5.2, h: 0.3, fontSize: 12, color: C.accent5 });
  steps(s, [
    ["WHAT WE DID", "Re-ran round, stretched and stretched + rotated blobs with 5 seeds and 3 budgets each"],
    ["WHY", "Akim had one run per setting; a single run can be luck"],
    ["RESULT", "+3.2 dB (22.3 → 25.5), seed SD ≤ 0.13 dB; learned blobs as stretched as real nuclei"],
  ]);
  notes(s,
    "Our first own step was to check Akim's shape result with repeats: the same three blob shapes, at 50, 100 and 250 blobs, each with 5 random seeds, on the same kind of real crops. The result holds at every budget: averaged over all 15 fits, 22.3 dB for round blobs and 25.5 for stretched and rotated. Repeats with different seeds differ by at most about a tenth of a dB, far less than the 3 dB effect. Most of the gain comes from stretching; rotation adds about 0.3 dB. And the blobs did what we expected: their depth-to-width ratio, 0.79, is close to the ratio measured on the real nuclei, 0.73. Real nuclei look stretched in depth partly because the microscope blurs more along that axis.",
    ["Round (isotropic) blob: same size in every direction.",
     "Stretched (anisotropic, axis-aligned) blob: three independent sizes along x, y and z.",
     "Rotation: lets the stretched blob point in any direction.",
     "Seed: the random number that decides the starting point of a fit; different seeds test whether a result is luck.",
     "SD (standard deviation): the typical distance of single results from their average; here, how much repeats differ."],
    ["Did this also help finding nuclei? Slightly: nucleus F1 0.745 (round) vs 0.805 (stretched + rotated), significant in a paired test (p = 0.006, 5 seeds).",
     "Why does the axis start at 20? To make a 3 dB difference visible; it is stated on the chart."]);
}

// ================================================================ 7. step 2 seeding
{
  const s = content("Simple setup and changes", "Step 6 · Start each blob on a nucleus");
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
  const s = content("Simple setup and changes", "Step 7 · Bigger steps let blobs travel");
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
    ["Does that undo Steps 1 and 6? It reframes them: starting points mattered so much because blobs could not move. With a suitable step size, a random start reaches the same F1."]);
}

// ================================================================ 9. step 4 size cap
{
  const s = content("Simple setup and changes", "Step 8 · Faint nuclei: cap the blob size");
  s.addImage({ path: FIG("size_cap.png"), x: 0.5, y: 1.2, w: 5.2, h: 2.6, objectName: "size-cap" });
  txt(s, "Green: nucleus found  ·  red: missed", { x: 0.5, y: 3.9, w: 5.2, h: 0.3, fontSize: 11, color: C.accent5 });
  steps(s, [
    ["WHAT WE CHANGED", "Limited how wide a nucleus blob may grow (10 voxels)"],
    ["WHY", "At the end of fitting, blobs at lost faint nuclei were ~30% wider than at kept ones"],
    ["RESULT", "Faint nuclei found: 13 → 19 of 31. Image quality unchanged; 1 of 4 regions worse"],
  ]);
  notes(s,
    "We then asked which nuclei are still lost. They are the faint ones. At the end of fitting, the blob nearest to each lost faint nucleus was not dimmer than for the nuclei we kept, but about 30 percent wider, and a wide blob leaves only a flat peak for the detector. So we capped the width of nucleus blobs. With the cap, faint nuclei found went from 13 of 31 to 19 of 31 over four regions, and image quality changed by only 0.01 dB. Honestly: this is one seed, one region got worse, and on the Drosophila data the cap changed nothing. Also, we compared the blobs at the end of fitting; we did not track them through fitting, so the widening is an association that the cap result supports, not a demonstrated cause.",
    ["Faint nucleus: one whose brightness above the surroundings is in the lower half.",
     "Size cap: an upper limit on a blob's size, applied only to the blobs that started on nuclei.",
     "Region: one 64 x 128 x 128 crop of the Tribolium frame."],
    ["Is +6 nuclei significant? It is one seed and 4 regions, so treat it as promising, not settled. The result survived a scoring-bug fix (it was +7 before).",
     "Did you watch the blobs widen? No. We compared the blob nearest to each nucleus at the end of fitting, not blobs tracked over time. That width comparison was also scored before the scoring-bug fix and could not be redone."]);
}

// ================================================================ 10. naive vs final
addSection("Results and validation");
{
  const s = content("Results and validation", "From the simple setup to the final recipe");
  const head = ["Change", "By", "Measured on", "Before", "After"].map((t) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 } } }));
  const body = [
    ["Start on bright peaks", "Akim", "PSNR, phantom", "32.0 dB", "53.1 dB"],
    ["Stretched + rotated blobs", "Akim", "PSNR, real data, median", "20.4 dB", "24.0 dB"],
    ["Grow blobs during fitting", "Akim", "PSNR, ~35-50 blobs", "23.8 dB", "24.3 dB"],
    ["More blobs, in cell-count units", "Akim", "cell F1, 301 → 1,804 blobs", "0.35", "0.76"],
    ["Repeats of the shape test", "us", "PSNR, 15 fits", "22.3 dB", "25.5 dB"],
    ["One start per nucleus", "us", "nuclei found, of 110", "52%", "71%"],
    ["Bigger position steps", "us", "F1, random starts", "0.64", "0.78"],
    ["Size cap", "us", "faint nuclei found, of 31", "13", "19"],
  ].map((r) => r.map((t, j) => ({ text: t, options: { bold: j === 4, color: j === 4 ? HEX.accent1 : HEX.dk1, align: j >= 3 ? "center" : "left" } })));
  s.addTable([head].concat(body), { x: 0.5, y: 1.15, w: 9.0, colW: [2.75, 0.75, 2.6, 1.45, 1.45], fontSize: 14, fontFace: "Calibri",
    rowH: 0.36, border: { type: "solid", pt: 0.5, color: "E5E7EB" }, fill: { color: HEX.lt1 } });
  txt(s, "Separate comparisons under the stated settings; the numbers are not added together.",
    { x: 0.5, y: 4.62, w: 9.0, h: 0.4, fontSize: 14, color: C.text2 });
  notes(s,
    "Here are all eight changes side by side, four by Akim and four by us. I want to be clear that these are separate comparisons, each under the settings on its own slide, not a chain where every row builds on the one above. They used different data, budgets and numbers of repeats, so you cannot add them up into one overall improvement. One row needs care: the cell-count row shows F1 rising as the budget grows from 301 to 1,804 blobs; it does not show that choosing the budget from the cell count beats another way of choosing it. Together, the changes turn a fit that merely looks right into one where the nuclei are found much more reliably.",
    ["Naive / simple setup: round blobs with random starting positions, default step size, no cap."],
    ["Why not one combined number? Each experiment used a different number of regions, seeds and references; combining them would hide that."]);
}

// ================================================================ 11. validation
{
  const s = content("Results and validation", "Does the fit behave as expected?");
  const cards = [
    ["0.79 vs 0.73", "depth-to-width ratio: learned blobs vs measured nuclei", C.accent3],
    ["14 vs 10", "of 29 hand-marked Drosophila nuclei in the top 100 detections: fitted blobs vs raw brightness", C.accent3],
    ["16–18", "same top 100, raw candidates ranked by a simple LoG blob filter: better than every fit", C.accent4],
  ];
  cards.forEach(([big, small, col], i) => {
    const x = 0.5 + i * 3.05;
    card(s, x, 1.25, 2.85, 2.6, `check-${i}`);
    txt(s, big, { x: x + 0.2, y: 1.45, w: 2.45, h: 0.8, fontSize: 32, bold: true, color: col, fontFace: "Cambria" });
    txt(s, small, { x: x + 0.2, y: 2.35, w: 2.45, h: 1.4, fontSize: 15, color: C.text2 });
  });
  txt(s, "In the top 100, fitting ranks nuclei better than brightness but not better than a simple blob filter. With all 457 candidates the raw detector finds 26 of 29.",
    { x: 0.5, y: 4.1, w: 9.0, h: 0.7, fontSize: 15, color: C.text2 });
  notes(s,
    "Checks that the fit does what we think. First, shape: the learned blobs are about as stretched as the real nuclei. Second, the honest test against people's annotations: on Drosophila, among the 100 most confident detections, the fitted blobs contain 14 of 29 hand-marked nuclei, the raw image ranked by brightness only 10. So fitting ranks real nuclei higher than brightness does. But third, the stronger baseline: a classic blob filter, LoG, applied to the same raw candidates puts 16 to 18 marked nuclei in the top 100, in seconds and without any fitting. So the fit beats the naive baseline, not the simple strong one. And if the plain detector may use all its candidates it finds 26 of 29, more than any fit. This is better ranking than brightness, not a better detector.",
    ["Top 100 detections: compare methods at the same number of guesses, because only some nuclei are marked and extra detections cannot be judged as wrong.",
     "Raw brightness: the raw image's candidate peaks ranked simply by how bright they are.",
     "LoG blob filter: ranks the same candidates by how well they look like a round blob of nucleus size."],
    ["How noisy is 14 vs 10? One nucleus is 3.4% of 29; we only call a difference real if it is about 3 nuclei and has the same sign in all seeds (mean of 3 seeds here).",
     "Why 16-18? 16 with LoG scales fixed in advance (1.5-3 micrometres); 18 with a single 2-micrometre scale chosen after seeing results.",
     "And in the top 200? There the best fit (uniform sampling, 22.7) and LoG (22) are level, within one nucleus. The filter is clearly better only in the top 100.",
     "Are fits repeatable? Our own fits: seed spread at most 0.13 dB (Step 5). Luxar, used on the next slides: three repeated fits of one frame differed by at most 2.1-3.7 percentage points of nuclei kept, depending on the detector."]);
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
    if (f !== "raw") txt(s, [{ text: `${L[f][1]}% of nuclei kept`, options: { breakLine: true } },
      { text: "2D Cellpose, whole frame", options: { fontSize: 10 } }],
    { x, y: 3.15, w: 2.2, h: 0.45, fontSize: 12, color: C.accent5, align: "center" });
  });
  const pts = [["Luxar (2026)", "state-of-the-art Gaussian-blob tool for microscopy"], ["JPEG2000", "standard image compression, set to exactly the same file size"], ["Test", "12 C. elegans frames from 2 embryos, 2,958 hand-marked nuclei, 4 detectors"]];
  pts.forEach(([a, b], i) => {
    txt(s, [{ text: a, options: { bold: true, color: HEX.accent1, breakLine: true } }, { text: b }],
      { x: 0.5 + i * 3.05, y: 3.75, w: 2.85, h: 1.3, fontSize: 15, color: C.text1 });
  });
  notes(s,
    "Our own fits were better than our naive start, but are they better than what exists? The strongest existing tool is Luxar, from the Royer lab, which fits Gaussian blobs to large microscopy data. We compared Luxar with ordinary image compression, JPEG2000, squeezed to exactly the same number of bytes, on 12 C. elegans frames from two embryos with almost 3,000 hand-marked nuclei, and asked four different nucleus detectors how many nuclei survive. The pictures show the same region at about 89 times smaller: the codecs keep the texture, Luxar draws smooth blobs. The percentages are whole-frame scores from one detector, 2D Cellpose.",
    ["Compression ratio: original size divided by compressed size; 100x means 100 times smaller.",
     "Same file size (matched bytes): every Luxar file is compared with a JPEG2000 file of exactly the same size on disk.",
     "JPEG2000 / JPEG-XL: standard image compression formats; here each 2D slice of the volume is compressed.",
     "Nuclei kept: hand-marked nuclei found after compression divided by those found in the original by the same detector. 1.0 means the same number was found, not necessarily the same individual nuclei.",
     "Whole-frame score: the percentages under the pictures are for the whole frame with 2D Cellpose; the pictures show only a small region."],
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
    title: "Luxar minus JPEG2000: nuclei kept (points), mean of 12 frames, 2 embryos", titleFontSize: 12, titleColor: HEX.dk2, barGapWidthPct: 40 }));
  card(s, 6.35, 1.25, 3.15, 1.55, "psnr-card");
  txt(s, [{ text: "PSNR: JPEG2000 better", options: { bold: true, breakLine: true } }, { text: "on 12 of 12 frames" }],
    { x: 6.55, y: 1.42, w: 2.8, h: 1.2, fontSize: 18, color: C.text1 });
  card(s, 6.35, 2.95, 3.15, 1.95, "detectors-card");
  txt(s, [{ text: "Detectors disagree", options: { bold: true, breakLine: true } },
    { text: "Cellpose 3D prefers blobs; watershed and Cellpose 2D prefer JPEG2000; all prefer JPEG2000 past 300×" }],
  { x: 6.55, y: 3.1, w: 2.8, h: 1.7, fontSize: 15, color: C.text1 });
  notes(s,
    "The result. Bars above zero mean the blobs kept more nuclei than JPEG2000 at the same file size; below zero, JPEG2000 kept more. Up to about 300 times compression the answer depends on the detector: Cellpose in 3D mode finds 3.5 points more nuclei in the blob version, the classical watershed and Cellpose in 2D mode find more in JPEG2000, and the simple blob detector LoG shows no consistent difference. Beyond 300 times, every detector prefers JPEG2000 by 17 to 23 points. Meanwhile PSNR says JPEG2000 is better on every single frame, so PSNR alone could not have told you which format a given analysis needs. The same Cellpose network changing sides between 2D and 3D mode supports an explanation involving processing geometry: slice by slice versus the whole volume. Switching modes shows that the behaviour differs; it does not isolate the exact cause.",
    ["LoG (Laplacian of Gaussian): a classic blob detector that responds to bright round spots of a given size.",
     "Watershed: a classical segmentation that separates touching bright regions, like water filling valleys.",
     "Cellpose: a widely used deep-learning model that outlines cells or nuclei. 2D mode works slice by slice and joins slices; 3D mode looks at the volume in three directions at once.",
     "Points: percentage points of nuclei kept, e.g. 0.90 vs 0.86 is 4 points.",
     "Statistics: each frame counts once; Wilcoxon signed-rank tests over the 12 frames, Holm-corrected for 8 tests. Everything except LoG below 300x is significant (Holm-corrected p ≤ 0.01). The 12 frames come from only 2 embryos, so they are not 12 independent animals."],
    ["Is LoG +1.3 real? No: 8 of 12 frames, p = 0.11; it comes from one embryo only.",
     "Is slice-by-slice processing proven to be the cause? No. Same network, same weights, different behaviour: so the penalty is not a fixed property of the learned model. But 3D mode also changes how predictions are combined and cleaned up, so the mode switch does not isolate geometry as the cause.",
     "Why frames and not single comparisons? An earlier draft averaged all matched pairs, with bootstrap intervals per pair. Pairs from one frame are not independent, so the final analysis averages per frame first and tests across frames; the numbers here are that final analysis, the same as in the paper.",
     "Why does JPEG2000 win at high compression? Below about 3 blobs per nucleus, blob fits lose whole nuclei; wavelet compression degrades gradually."]);
}

// ================================================================ 14. cliff
{
  const s = content("State of the art", "C. elegans: nuclei vanish below ~3 blobs each");
  s.addImage({ path: FIG("cliff_deck.png"), x: 0.5, y: 1.2, w: 5.75, h: 3.4, objectName: "cliff" });
  steps(s, [
    ["WHAT WE SEE", "Under LoG, nuclei kept crosses 90% at 1.1–3.1 blobs per nucleus (all 12 frames)"],
    ["PSNR", "For budgets of 1,000+ blobs it moves at most 2.6 dB per frame"],
    ["SO", "Choosing the budget by PSNR can miss the cliff: check the nuclei"],
  ], 6.45, 3.05);
  notes(s,
    "Why do blobs collapse at high compression? Each point is one Luxar fit; the x axis is blobs per hand-marked nucleus. In these C. elegans frames, with the LoG detector, almost every nucleus survives above about 3 blobs per nucleus; below that, they disappear quickly, on every frame of both embryos: the 90% line is crossed between 1.1 and 3.1 blobs per nucleus. For budgets of 1,000 blobs and more, PSNR changes by at most 2.6 dB per frame. So if you choose the budget by image quality, you can fall off this cliff without noticing. Honestly, our advance test of that PSNR claim held on only 4 of the 8 added frames, because on small frames the cliff lies below 1,000 blobs.",
    ["Blobs per nucleus: number of blobs in the fit divided by the number of hand-marked nuclei in that frame.",
     "Cliff: a sudden drop instead of a gradual decline.",
     "90% line: the threshold we fixed in advance to define the cliff (we wrote down '≤ 3.5 blobs per nucleus' before running the extra frames)."],
    ["Does the 3-blobs rule hold everywhere? We only tested C. elegans frames with the LoG detector; other data or detectors may cross elsewhere.",
     "Why fix thresholds in advance? So we cannot tune the claim to the data afterwards. Two of our advance predictions failed, and we report them as failed."]);
}

// ================================================================ 15. conclusion
addSection("Conclusion");
{
  const s = pres.addSlide({ masterName: "DARK_LIST", sectionTitle: "Conclusion" });
  n += 1;
  currentTitle = "What we learned";
  s.addText(currentTitle, { placeholder: "title" });
  s.addText([
    { text: "Stretched blobs, one start per nucleus and a size cap made our fits find more nuclei; in the top 100, a simple blob filter still ranks them better.", options: { bullet: true, breakLine: true } },
    { text: "At the same file size, the better format depends on the detector and the compression: blobs help 3D Cellpose below 300×, JPEG2000 wins beyond.", options: { bullet: true, breakLine: true } },
    { text: "Viewing and measuring need separate checks: PSNR cannot choose the format.", options: { bullet: true } },
  ], { placeholder: "body", paraSpaceAfter: 12 });
  s.addImage({ path: FIG("title_tribolium.png"), x: 7.05, y: 0.35, w: 2.55, h: 4.85, objectName: "end-tribolium" });
  notes(s,
    "Three take-aways. One: our step-by-step changes made blob fits find nuclei more reliably, but on hand-marked nuclei a simple blob filter still ranks them better among the top 100 detections, and as well among the top 200. Two: compared with the state of the art at the same file size, there is no single winner. Below 300 times, the blobs help 3D Cellpose and hurt the watershed and 2D Cellpose; beyond 300 times JPEG2000 keeps more nuclei for every detector. Three: a format that looks good is not automatically good for measuring, and the usual image-quality number cannot decide this; you have to test with the analysis you plan to run.",
    ["Codec: a compression method (coder-decoder)."],
    ["So was the project a failure? No: we now know where blobs should and should not be trusted, which is what a user needs; and the evaluation method itself is reusable.",
     "Are blobs useless for measuring? Not in general: with 3D Cellpose they kept more nuclei at moderate compression. But each analysis has to be checked separately."]);
}

// ================================================================ 16. next
{
  const s = content("Conclusion", "Next: compression that biologists can use");
  const items = [
    ["Lossless storage", "what labs keep today: bit-exact; 4.2× with lossless JPEG-XL on one C. elegans frame"],
    ["Noise-bounded coding", "one specific near-lossless method: every change kept below the camera noise"],
    ["Video codecs", "use the similarity between neighbouring slices and time points"],
    ["Judge by tracking", "measure with the Cell Tracking Challenge scores, not image quality"],
  ];
  items.forEach(([a, b], i) => {
    const y = 1.15 + i * 0.95;
    s.addShape(pres.shapes.OVAL, { x: 0.5, y: y + 0.05, w: 0.55, h: 0.55, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: `num-${i}` });
    txt(s, String(i + 1), { x: 0.5, y: y + 0.05, w: 0.55, h: 0.55, fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle" });
    txt(s, [{ text: a, options: { bold: true, breakLine: true } }, { text: b, options: { color: HEX.dk2 } }],
      { x: 1.3, y, w: 8.2, h: 0.85, fontSize: 16 });
  });
  notes(s,
    "This is a change of direction, agreed with our supervisor after his feedback. So far we asked whether Gaussian blobs can store and show microscopy data; he pointed out that labs store their raw data losslessly and rarely use lossy formats like JPEG. So the next step is compression biologists would actually use. First, lossless storage, which gives about 2 to 4 times; we measured 4.2 times with lossless JPEG-XL on one C. elegans frame. Second, one specific near-lossless approach, noise-bounded coding, where every voxel may change but by less than the camera noise. Third, video codecs, which exploit the similarity between neighbouring slices and time points. And fourth, judging success by whether cells can still be tracked, using the Cell Tracking Challenge's official scores.",
    ["Lossless: the decompressed data is bit-for-bit identical to the original.",
     "Near-lossless: small, bounded changes are allowed. Noise-bounded coding is one way to set that bound: below the camera's own noise.",
     "Video codec: compression made for movies (e.g. H.265, AV1) that predicts each frame from its neighbours.",
     "Tracking: following each cell over time, including divisions."],
    ["How much do lossless methods give? On our C. elegans frame: lossless JPEG-XL 4.2x, JPEG-LS 2.9x, JPEG2000 lossless 2.7x, Zstandard 2.5x. Noise-bounded coding can go further; how far depends on the camera noise, and we have not measured it on our data yet.",
     "Why change direction? The supervisor's feedback: biologists need their measurements to stay valid, and they trust lossless or noise-bounded storage. Gaussian blobs remain interesting for viewing."]);
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
