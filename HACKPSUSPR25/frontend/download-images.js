import fs from 'fs';
import https from 'https';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const letters = [
  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
  'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
  'U', 'V', 'W', 'X', 'Y', 'Z'
];

const baseUrl = 'https://raw.githubusercontent.com/dmyersturnbull/asl-signs/main/letters';
const outputDir = path.join(__dirname, 'public', 'asl_letters');

// Create output directory if it doesn't exist
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true });
}

// Download each letter image
letters.forEach(letter => {
  const fileName = `${letter.toLowerCase()}.png`;
  const filePath = path.join(outputDir, fileName);
  const url = `${baseUrl}/${fileName}`;

  https.get(url, (response) => {
    if (response.statusCode === 200) {
      const fileStream = fs.createWriteStream(filePath);
      response.pipe(fileStream);

      fileStream.on('finish', () => {
        console.log(`Downloaded ${fileName}`);
        fileStream.close();
      });
    } else {
      console.error(`Failed to download ${fileName}: ${response.statusCode}`);
    }
  }).on('error', (err) => {
    console.error(`Error downloading ${fileName}: ${err.message}`);
  });
}); 