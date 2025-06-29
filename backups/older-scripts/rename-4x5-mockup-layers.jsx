// ====================================================
// 📄 Rename Only Visible Layers to "4x5-mockup-01" etc.
// ✅ Works inside groups too
// 🛠️ Usage: File > Scripts > Browse... in Photoshop
// ====================================================

function renameVisibleLayers(prefix) {
    if (!app.documents.length) {
        alert("No document open!");
        return;
    }

    var doc = app.activeDocument;
    var count = 1;

    function pad(num, size) {
        var s = "00" + num;
        return s.substr(s.length - size);
    }

    function processLayer(layer) {
        if (!layer.visible) return; // Skip hidden layers

        if (layer.typename === "ArtLayer") {
            layer.name = prefix + pad(count, 2);
            count++;
        } else if (layer.typename === "LayerSet") {
            for (var i = 0; i < layer.layers.length; i++) {
                processLayer(layer.layers[i]);
            }
        }
    }

    // Reverse order for correct stacking
    for (var i = doc.layers.length - 1; i >= 0; i--) {
        processLayer(doc.layers[i]);
    }

    alert("✅ Visible layers renamed as '" + prefix + "##'");
}

// Run it with your desired prefix
renameVisibleLayers("4x5-mockup-");
