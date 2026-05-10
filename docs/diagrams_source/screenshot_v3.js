const puppeteer = require("puppeteer");
const path = require("path");

(async () => {
  const browser = await puppeteer.launch({ headless: "new" });
  const page = await browser.newPage();
  await page.setViewport({ width: 1540, height: 1700, deviceScaleFactor: 2 });
  const fileUrl = "file:///" + path.resolve("pipeline_v3.html").replace(/\\/g, "/");
  await page.goto(fileUrl, { waitUntil: "networkidle0" });
  await new Promise(r => setTimeout(r, 500));
  const dim = await page.evaluate(() => {
    const c = document.querySelector(".frame");
    const r = c.getBoundingClientRect();
    return { w: Math.ceil(r.right + 30), h: Math.ceil(r.bottom + 30) };
  });
  await page.setViewport({ width: dim.w, height: dim.h, deviceScaleFactor: 2 });
  await new Promise(r => setTimeout(r, 300));
  await page.screenshot({ path: "pipeline_v3.png", fullPage: true });
  await browser.close();
  console.log(`SAVED pipeline_v3.png ${dim.w * 2}x${dim.h * 2}`);
})();
