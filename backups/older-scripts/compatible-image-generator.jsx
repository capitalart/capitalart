// ===================================================
// Photoshop JSX Script: Export JPEG (Harvey Norman Safe)
// ===================================================

// Set export folder (same as current file)
var doc = app.activeDocument;
var exportFolder = doc.path;

// Set output name
var fileName = doc.name.replace(/\.[^\.]+$/, '') + "_HN-safe.jpg";
var saveFile = new File(exportFolder + "/" + fileName);

// JPEG export options
var jpegOptions = new JPEGSaveOptions();
jpegOptions.quality = 9; // Max quality
jpegOptions.embedColorProfile = true;
jpegOptions.formatOptions = FormatOptions.STANDARDBASELINE; // Not Progressive
jpegOptions.matte = MatteType.NONE;

// Export
doc.saveAs(saveFile, jpegOptions, true);

// Alert when done
alert("✅ Export complete:\n" + fileName + "\nReady for Harvey Norman printing.");
