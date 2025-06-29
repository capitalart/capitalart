#target photoshop

function zeroPad(n, width) {
    width = width || 2;
    n = n + '';
    return n.length >= width ? n : new Array(width - n.length + 1).join('0') + n;
}

var doc = app.activeDocument;
var outputFolder = Folder.selectDialog("Select folder to save artwork layers");

if (outputFolder == null) {
    alert("Export cancelled.");
} else {
    for (var i = 0; i < doc.layerSets.length; i++) {
        var group = doc.layerSets[i];
        var groupName = group.name;
        var index = zeroPad(i + 1); // Starts at 01

        // Hide all groups
        for (var j = 0; j < doc.layerSets.length; j++) {
            doc.layerSets[j].visible = false;
        }

        group.visible = true;

        // Save as PNG
        var saveFile = new File(outputFolder + "/artwork-layer-" + index + ".png");
        var opts = new PNGSaveOptions();
        opts.compression = 9;
        doc.saveAs(saveFile, opts, true, Extension.LOWERCASE);
    }

    alert("✅ All groups exported as PNGs!");
}
