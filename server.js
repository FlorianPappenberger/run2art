/**
 * server.js — GPS Art Node.js Orchestrator
 *
 * Serves the static frontend and proxies /api/match requests
 * to the Python geospatial engine (engine.py).
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

const PORT = process.env.PORT || 3000;

// MIME types for static files
const MIME = {
  '.html': 'text/html',
  '.css': 'text/css',
  '.js': 'application/javascript',
  '.json': 'application/json',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

// ── Static file server ─────────────────────────────────
function serveStatic(req, res) {
  let filePath = path.join(__dirname, 'public', req.url === '/' ? 'index.html' : req.url);
  const ext = path.extname(filePath).toLowerCase();
  const mime = MIME[ext] || 'application/octet-stream';

  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end('Not found');
      return;
    }
    res.writeHead(200, { 'Content-Type': mime });
    res.end(data);
  });
}

// ── Python engine caller ───────────────────────────────
function callEngine(payload) {
  return new Promise((resolve, reject) => {
    const py = spawn('python', [path.join(__dirname, 'engine.py')], {
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';

    py.stdout.on('data', chunk => { stdout += chunk; });
    py.stderr.on('data', chunk => { stderr += chunk; });

    py.on('close', code => {
      if (code !== 0) {
        console.error('[engine.py stderr]', stderr);
        try {
          // engine may return JSON error on stderr too
          const errJson = JSON.parse(stdout);
          resolve(errJson);
        } catch {
          reject(new Error(`engine.py exited with code ${code}: ${stderr}`));
        }
        return;
      }
      try {
        resolve(JSON.parse(stdout));
      } catch {
        reject(new Error('Invalid JSON from engine.py: ' + stdout.slice(0, 300)));
      }
    });

    py.on('error', err => reject(err));

    // Send payload to engine via stdin
    py.stdin.write(JSON.stringify(payload));
    py.stdin.end();
  });
}

// ── HTTP Server ────────────────────────────────────────
const server = http.createServer(async (req, res) => {
  // API: POST /api/match
  if (req.method === 'POST' && req.url === '/api/match') {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', async () => {
      try {
        const payload = JSON.parse(body);
        console.log(`[api/match] mode="${payload.mode || 'fit'}" shape="${payload.shape_name || payload.shapes?.[payload.shape_index]?.name || '?'}" center=${JSON.stringify(payload.center_point)}`);
        const result = await callEngine(payload);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(result));
      } catch (err) {
        console.error('[api/match error]', err.message);
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: err.message }));
      }
    });
    return;
  }

  // Everything else → static files
  serveStatic(req, res);
});

server.listen(PORT, () => {
  console.log(`\n  GPS Art server running at  http://localhost:${PORT}\n`);
});
