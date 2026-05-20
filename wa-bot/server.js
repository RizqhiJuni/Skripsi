/**
 * Mini WhatsApp bridge berbasis whatsapp-web.js.
 * Mengekspos endpoint yang KOMPATIBEL dengan WAHA, sehingga alarm.py
 * tidak perlu diubah:
 *
 *   GET  /                               -> halaman QR + status
 *   GET  /api/sessions/:session          -> { status: "WORKING" | ... }
 *   POST /api/sendText                   -> { chatId, text }
 *   POST /api/sendImage                  -> { chatId, file:{data,mimetype,filename}, caption }
 *
 * Jalankan:
 *   cd wa-bot
 *   npm install
 *   npm start
 *
 * Lalu buka http://localhost:3000 untuk scan QR.
 */

const express = require('express');
const QRCode = require('qrcode');
const qrcodeTerminal = require('qrcode-terminal');
const { Client, LocalAuth, MessageMedia } = require('whatsapp-web.js');

const PORT = process.env.PORT || 3000;
const app = express();
app.use(express.json({ limit: '25mb' }));

// ---------------------------------------------------------------- WA client
let latestQr = null;
let status = 'STARTING'; // STARTING | SCAN_QR_CODE | WORKING | FAILED

const client = new Client({
  authStrategy: new LocalAuth({ dataPath: './.wwebjs_auth' }),
  puppeteer: {
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
  },
});

client.on('qr', (qr) => {
  latestQr = qr;
  status = 'SCAN_QR_CODE';
  console.log('\n[WA] Scan QR di http://localhost:' + PORT + ' (atau terminal):');
  qrcodeTerminal.generate(qr, { small: true });
});

client.on('authenticated', () => {
  console.log('[WA] Authenticated.');
  status = 'AUTHENTICATED';
});

client.on('ready', () => {
  console.log('[WA] Client READY. Bot siap menerima request.');
  latestQr = null;
  status = 'WORKING';
});

client.on('auth_failure', (msg) => {
  console.error('[WA] Auth failure:', msg);
  status = 'FAILED';
});

client.on('disconnected', (reason) => {
  console.warn('[WA] Disconnected:', reason);
  status = 'STARTING';
  client.initialize();
});

client.initialize();

// --------------------------------------------------------------------- utils
function normalizeChatId(id) {
  if (!id) throw new Error('chatId kosong');
  if (id.includes('@')) return id;
  const digits = String(id).replace(/\D/g, '');
  if (!digits) throw new Error('chatId tidak valid: ' + id);
  return digits + '@c.us';
}

function ensureReady(res) {
  if (status !== 'WORKING') {
    res.status(503).json({ error: 'WA belum siap', status });
    return false;
  }
  return true;
}

// ------------------------------------------------------------------- routes
app.get('/', async (req, res) => {
  let qrImg = '';
  if (latestQr) {
    try {
      qrImg = await QRCode.toDataURL(latestQr, { width: 320 });
    } catch (e) { /* ignore */ }
  }
  res.send(`<!doctype html>
<html><head><title>WA Bot</title>
<meta http-equiv="refresh" content="3"></head>
<body style="font-family:sans-serif;text-align:center;padding:24px">
  <h2>WhatsApp Bot</h2>
  <p>Status: <b>${status}</b></p>
  ${qrImg ? `<img src="${qrImg}" alt="QR"><p>Scan dengan HP pengirim alarm.</p>` : ''}
  ${status === 'WORKING' ? '<p>✅ Siap kirim pesan.</p>' : ''}
</body></html>`);
});

// WAHA-compatible status
app.get('/api/sessions/:session', (req, res) => {
  res.json({ name: req.params.session, status });
});

// WAHA-compatible: kirim teks
app.post('/api/sendText', async (req, res) => {
  if (!ensureReady(res)) return;
  try {
    const chatId = normalizeChatId(req.body.chatId);
    const text = String(req.body.text || '');
    const msg = await client.sendMessage(chatId, text);
    res.json({ ok: true, id: msg.id?._serialized });
  } catch (e) {
    console.error('sendText error:', e.message);
    res.status(500).json({ error: e.message });
  }
});

// WAHA-compatible: kirim gambar (base64)
app.post('/api/sendImage', async (req, res) => {
  if (!ensureReady(res)) return;
  try {
    const chatId = normalizeChatId(req.body.chatId);
    const file = req.body.file || {};
    if (!file.data) throw new Error('file.data (base64) kosong');
    const media = new MessageMedia(
      file.mimetype || 'image/jpeg',
      file.data,
      file.filename || 'image.jpg',
    );
    const caption = String(req.body.caption || '');
    const msg = await client.sendMessage(chatId, media, { caption });
    res.json({ ok: true, id: msg.id?._serialized });
  } catch (e) {
    console.error('sendImage error:', e.message);
    res.status(500).json({ error: e.message });
  }
});

app.listen(PORT, () => {
  console.log(`[WA] Bridge listening on http://localhost:${PORT}`);
  console.log('     Buka URL itu di browser untuk scan QR.');
});
