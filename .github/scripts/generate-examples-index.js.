// by Facundo Franchino
// Rebuild .github/examples_index.json by checking which Example files exist.
// - Reads current JSON (functions[] | items[] | []), preserves python names by default,
//   but can normalise names (ENV NORMALISE_NAMES=true).
// - Flips status to 'done' if the example exists somewhere in the repo
//   as either a .py *or* a .ipynb.
// - Sorts by MATLAB name and writes back prettified JSON.
// Sorts out CamelCase to snake_case naming convention differences between fdnToolbox and pyFDN.
// ENV: PY_ROOT (default: '' = whole repo; set to 'examples' if you want to limit the scan)
// ENV: NORMALISE_NAMES ("true"/"false") to rewrite python names to derived snake_case.

const fs = require('fs');
const path = require('path');

const INDEX_PATH = '.github/examples_index.json';
const PY_ROOT = process.env.PY_ROOT || ''; // search whole repo by default
const NORMALISE = String(process.env.NORMALISE_NAMES || 'false').toLowerCase() === 'true';

function walk(dir, exts, out = []) {
  if (!dir || !fs.existsSync(dir)) return out;
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, ent.name);
    if (ent.isDirectory()) walk(p, exts, out);
    else if (exts.includes(path.extname(ent.name).toLowerCase())) out.push(p);
  }
  return out;
}

// keep acronyms & digits glued; split only real camel humps and acronym→word boundaries.
function camelToSnakeBase(matlabName) {
  const base = matlabName.replace(/\.m$/i, '');
  return base
    // split between an acronym and the following Word (HTTPServer -> HTTP_Server)
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1_$2')
    // split lower/digit before Upper (detPolynomial -> det_Polynomial)
    .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
    // collapse underscores (safety)
    .replace(/_+/g, '_')
    .toLowerCase();
}

function readIndex() {
  if (!fs.existsSync(INDEX_PATH)) {
    throw new Error(`Missing ${INDEX_PATH}`);
  }
  const raw = fs.readFileSync(INDEX_PATH, 'utf8');
  const json = JSON.parse(raw);
  if (Array.isArray(json)) return json;
  if (Array.isArray(json.functions)) return json.functions;
  if (Array.isArray(json.items)) return json.items;
  throw new Error('examples_index.json must be an array, or have functions[] / items[]');
}

function writeIndex(items) {
  const out = { functions: items.sort((a, b) => a.matlab.localeCompare(b.matlab)) };
  const pretty = JSON.stringify(out, null, 2) + '\n';
  fs.writeFileSync(INDEX_PATH, pretty, 'utf8');
}

(function main() {
  // collect all .py and .ipynb paths once (case-insensitive existence set)
  const fileList = walk(PY_ROOT || process.cwd(), ['.py', '.ipynb']);
  const fileNames = new Set(fileList.map(p => path.basename(p).toLowerCase()));

  const items = readIndex();

  // simple normaliser helpers
  const norm = (s) => String(s || '').toLowerCase().replace(/_/g, '');
  const looksLikeLetterUnderscoreChain = (s) => /^[a-z](?:_[a-z])+\.(py)$/i.test(String(s || ''));

  let done = 0, todo = 0, touched = 0;

  const updated = items.map(it => {
    const matlab = it.matlab;

    // derive snake name from MATLAB (with improved acronym handling)
    const derivedBase = camelToSnakeBase(matlab);
    const derivedPy = `${derivedBase}.py`;
    const derivedNb = `${derivedBase}.ipynb`;

    // choose python filename to store in the index:
    // - by default keep existing if present
    // - if NORMALISE_NAMES=true, replace when:
    //    * there is no existing name, or
    //    * existing normalised equals derived normalised (e.g. e_d_c.py -> edc.py), or
    //    * existing looks like letter_letter_letter pattern (e.g. e_d_c.py)
    let python = (it.python && it.python.trim().length) ? it.python.trim() : derivedPy;
    if (NORMALISE) {
      if (!it.python || norm(it.python) === norm(derivedPy) || looksLikeLetterUnderscoreChain(it.python)) {
        python = derivedPy;
      }
    }

    // existence check (case-insensitive), consider both .py and .ipynb
    const chosenBase = path.basename(python).toLowerCase();
    const exists =
      fileNames.has(chosenBase) ||
      fileNames.has(path.basename(derivedPy).toLowerCase()) ||
      fileNames.has(path.basename(derivedNb).toLowerCase());

    const newStatus = exists ? 'done' : 'todo';
    if (newStatus === 'done') done++; else todo++;

    const changed =
      (String(it.status || '').toLowerCase() !== newStatus) ||
      (String(it.python || '') !== python);

    if (changed) touched++;

    return {
      matlab,
      python,
      status: newStatus,
      description: it.description || ''
    };
  });

  writeIndex(updated);

  console.log('examples_index scan summary:');
  console.log(`  example files found (.py/.ipynb): ${fileList.length}`);
  console.log(`  items: ${updated.length} (done: ${done}, todo: ${todo})`);
  console.log(`  changed entries: ${touched}`);
})();
