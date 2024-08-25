/** 一个像素的信息由 r, g, b 的值构成 */
export type PixelData = [number, number, number];
/** 一张图片的信息由若干像素点构成 */
export type ImgPixels = PixelData[];

import * as path from 'node:path';

// 獲取當前模組的 URL 並轉換為檔案路徑
let __filename = new URL(import.meta.url).pathname;

// 如果在 Windows 上運行，路徑會包含前導的斜杠（`/`），需要去掉
if (Deno.build.os === "windows") {
  __filename = __filename.substring(1);
}

// 獲取當前文件所在的目錄
const __dirname = path.dirname(__filename);

// export async function getDataList(): Promise<ImgPixels[]> {
//   const dataList: ImgPixels[] = [];
//   for await (const file of Deno.readDir('../imgs/test')){
//     if (file.isFile && file.name.endsWith('.json')) {
//       const json = await Deno.readTextFile(`../imgs/test/${file.name}`);
//       const data = JSON.parse(json) as ImgPixels;
//       dataList.push(data);
//     }
//   }

//   return dataList;
// }

export async function getDataList(): Promise<{ filenames: string[]; dataList: ImgPixels[] }> {
  const filenames: string[] = [];
  const dataList: ImgPixels[] = [];
  for await (const file of Deno.readDir(path.join(__dirname, `../../data/1_color_extraction_output/json`))){
    if (file.isFile && file.name.endsWith('.json')) {
      const json = await Deno.readTextFile(path.join(__dirname,`../../data/1_color_extraction_output/json/${file.name}`));
      const data = JSON.parse(json) as ImgPixels;
      filenames.push(file.name.split('.')[0]);
      dataList.push(data);
    }
  }

  return { filenames, dataList };
}

export const mainColorNumber = 7;

export function rgb2Hex(pixel: PixelData): string {
  return pixel.reduce((prevStr, channel) => {
    return prevStr + channel.toString(16).padStart(2, "0");
  }, "#");
}
