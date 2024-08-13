import { getDataList, mainColorNumber, PixelData, rgb2Hex } from "./shared.ts";
import * as fs from 'node:fs';

const img_pair = await getDataList();
const dataList = img_pair.dataList;
const filenames = img_pair.filenames


class Node {
  static leafNum = 0;
  static toReduce: Node[][] = new Array(8).fill(0).map(() => []);

  children: (Node | null)[] = new Array(8).fill(null);
  isLeaf = false;
  r = 0;
  g = 0;
  b = 0;
  childrenCount = 0;

  constructor(info?: { index: number; level: number }) {
    if (!info) return;
    if (info.level === 7) {
      this.isLeaf = true;
      Node.leafNum++;
    } else {
      Node.toReduce[info.level].push(this);
      Node.toReduce[info.level].sort(
        (a, b) => a.childrenCount - b.childrenCount
      );
    }
  }

  addColor(color: PixelData, level: number) {
    if (this.isLeaf) {
      this.childrenCount++;
      this.r += color[0];
      this.g += color[1];
      this.b += color[2];
    } else {
      let str = "";
      const r = color[0].toString(2).padStart(8, "0");
      const g = color[1].toString(2).padStart(8, "0");
      const b = color[2].toString(2).padStart(8, "0");

      str += r[level];
      str += g[level];
      str += b[level];
      const index = parseInt(str, 2);

      if (this.children[index] === null) {
        this.children[index] = new Node({
          index,
          level: level + 1,
        });
      }
      (this.children[index] as Node).addColor(color, level + 1);
    }
  }
}
function reduceTree() {
  // find the deepest level of node
  let lv = 6;

  while (lv >= 0 && Node.toReduce[lv].length === 0) lv--;
  if (lv < 0) return;

  const node = Node.toReduce[lv].pop() as Node;

  // merge children
  node.isLeaf = true;
  node.r = 0;
  node.g = 0;
  node.b = 0;
  node.childrenCount = 0;
  for (let i = 0; i < 8; i++) {
    if (node.children[i] === null) continue;
    const child = node.children[i] as Node;
    node.r += child.r;
    node.g += child.g;
    node.b += child.b;
    node.childrenCount += child.childrenCount;
    Node.leafNum--;
  }

  Node.leafNum++;
}

function colorsStats(node: Node, record: Record<string, number>) {
  if (node.isLeaf) {
    const r = (~~(node.r / node.childrenCount));
      // .toString(16)
      // .padStart(2, "0");
    const g = (~~(node.g / node.childrenCount));
      // .toString(16)
      // .padStart(2, "0");
    const b = (~~(node.b / node.childrenCount));
      // .toString(16)
      // .padStart(2, "0");

    // const color = "#" + r + g + b;
    const color = `[${r}, ${g}, ${b}]`;
    if (record[color]) record[color] += node.childrenCount;
    else record[color] = node.childrenCount;

    return;
  }

  for (let i = 0; i < 8; i++) {
    if (node.children[i] !== null) {
      colorsStats(node.children[i] as Node, record);
    }
  }
}

const t0 = performance.now();
dataList.forEach((data, index) => {
  const imgName = filenames[index];

  console.log(`\n*** processing img ` + imgName + `***\n`);
  const root = new Node();

  Node.toReduce = new Array(8).fill(0).map(() => []);
  Node.leafNum = 0;

  data.forEach((pixel, index) => {
    root.addColor(pixel, 0);

    while (Node.leafNum > 16) reduceTree();
  });

  const record: Record<string, number> = {};
  colorsStats(root, record);
  const result = Object.entries(record)
    .sort((a, b) => b[1] - a[1])
    .slice(0, mainColorNumber); // 只取 7 個顏色

  // output 到 terminal
  // console.log(`img_name = '${imgName}';`);
  // result.forEach((color, i) => {
  //   console.log(`manual_color_${i} = ${color[0]};`);
  // });

  //output 到 txt
  const outputFile = `../output/${imgName}.txt`;
  const output = [
                    `img_name = '${imgName}.png';`,
                    ...result.map((color, i) => `manual_color_${i} = ${color[0]};`),
                  ].join(' ');

  fs.writeFileSync(outputFile, output);
  console.log(`Output written to ${outputFile}`);

});
const t1 = performance.now();
console.log(`Call to doSomething took ${(t1 - t0)/1000} seconds.`);

//img_name = 'Kim_Jisoo.jpg'; manual_color_0 = [176, 202, 211]; manual_color_1 = [87, 52, 42]; manual_color_2 = [27, 23, 23]; manual_color_3 = [114, 145, 145]; manual_color_4 = [210, 164, 149]; manual_color_5 = [172, 114, 88]; manual_color_6 = [229, 214, 205]; 