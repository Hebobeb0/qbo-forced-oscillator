const fs = require('fs');
const path = require('path');
const root = __dirname;
const out = path.join(root, '..', 'outputs', process.argv[2] || 'conclusion_graphs');
fs.mkdirSync(path.join(root, 'fontcache'), { recursive: true });
const config = path.join(root, 'fonts.xml');
fs.writeFileSync(config, `<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd"><fontconfig><dir>C:/Windows/Fonts</dir><cachedir>${path.join(root,'fontcache').replace(/\\/g,'/')}</cachedir></fontconfig>`);
process.env.FONTCONFIG_FILE = config;
const sharp = require('C:/Users/Surface/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp');
(async () => {
  for (const file of fs.readdirSync(out).filter(f => f.endsWith('.svg'))) {
    await sharp(path.join(out, file), { density: 144 }).flatten({ background: '#ffffff' }).png().toFile(path.join(out, file.replace('.svg', '.png')));
  }
  process.stdout.write('Rendered all six graphs.');
})();
