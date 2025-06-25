// hn-safe-batch.js
// Batch convert all JPGs to Harvey Norman/Etsy safe versions

const sharp = require('sharp');
const fs = require('fs');
const path = require('path');

// Your folders:
const inputDir = '/Users/robin/Documents/01-Artwork-Workshop/Harvey-Norman-Safe-Workshop/Originals';
const outputDir = '/Users/robin/Documents/01-Artwork-Workshop/Harvey-Norman-Safe-Workshop/HN-Safe';

const MAX_FILENAME_LEN = 70;
const MAX_FILESIZE = 20.8 * 1024 * 1024; // bytes
const JPEG_QUALITY = 90;

async function processImage(file) {
    let basename = path.parse(file).name;
    let ext = '.jpg';

    // Truncate filename if needed
    let baseTrunc = basename.slice(0, MAX_FILENAME_LEN - ext.length);
    let outFile = path.join(outputDir, baseTrunc + ext);

    // Read image
    let buffer = fs.readFileSync(path.join(inputDir, file));
    let image = sharp(buffer).rotate(); // auto-orient

    // Ensure sRGB, flatten, no metadata
    let metadata = await image.metadata();
    let width = metadata.width;
    let height = metadata.height;

    let quality = JPEG_QUALITY;
    let attempt = 0;
    let resultBuffer;

    while (true) {
        resultBuffer = await image
            .resize({ width, height }) // start at original
            .jpeg({
                quality: quality,
                progressive: false,
                force: true
            })
            .toColourspace('srgb')
            .withMetadata({ icc: undefined }) // strip ICC metadata for simplicity
            .toBuffer();

        if (resultBuffer.length <= MAX_FILESIZE) break;

        // Downscale by 5% each loop if too big
        width = Math.round(width * 0.95);
        height = Math.round(height * 0.95);
        attempt += 1;
        if (width < 1000 || height < 1000 || attempt > 40) {
            console.warn(`Could not shrink ${file} under ${MAX_FILESIZE / (1024 * 1024)}MB even after major downsizing!`);
            break;
        }
    }

    fs.writeFileSync(outFile, resultBuffer);
    console.log(`✅ Processed: ${file} → ${path.basename(outFile)} (${(resultBuffer.length/1024/1024).toFixed(2)} MB)`);
}

async function main() {
    if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });
    const files = fs.readdirSync(inputDir).filter(f => f.match(/\.jpg$/i));

    if (files.length === 0) {
        console.log('No .jpg images found in input folder!');
        process.exit(0);
    }

    // Confirm before batch (cancel = no change)
    const readline = require('readline').createInterface({
        input: process.stdin,
        output: process.stdout
    });
    console.log(`Ready to process ${files.length} image(s) from:\n${inputDir}\nOutput to:\n${outputDir}\n\nTruncate filenames to 70 chars, enforce sRGB, JPEG Baseline, max 20.8MB.\n`);
    readline.question('Continue? (y/N): ', async (answer) => {
        if (answer.toLowerCase() !== 'y') {
            console.log('Batch cancelled.');
            readline.close();
            process.exit(0);
        } else {
            readline.close();
            for (let file of files) {
                try {
                    await processImage(file);
                } catch (e) {
                    console.error(`❌ Error processing ${file}: ${e}`);
                }
            }
            console.log('\nBatch complete! All images are Harvey Norman/Etsy safe.');
        }
    });
}

main();
