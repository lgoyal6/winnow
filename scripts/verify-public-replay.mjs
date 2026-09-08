import { readFile } from "node:fs/promises";
const source=JSON.parse(await readFile("web/public/fixtures/demo-transcript.json","utf8"));
const published=JSON.parse(await readFile("public-proof/demo-transcript.json","utf8"));
if(JSON.stringify(source)!==JSON.stringify(published))throw Error("published fixture differs from RecordedSource fixture");
if(source.length!==13)throw Error(`expected 13 recorded utterances, found ${source.length}`);
if(source.some((row,index)=>index>0&&row.startMs<source[index-1].endMs))throw Error("fixture timing overlaps or runs backward");
if(source.some(row=>typeof row.text!=="string"||!row.text.trim()))throw Error("fixture contains an empty utterance");
console.log(`public replay valid: ${source.length} utterances, ${source.at(-1).endMs-source[0].startMs} ms span`);
