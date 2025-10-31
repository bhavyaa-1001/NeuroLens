import express from 'express';
import cors from 'cors';
import morgan from 'morgan';
import multer from 'multer';
import mongoose from 'mongoose';
import dotenv from 'dotenv';
import path from 'path';
import fs from 'fs';
import AdmZip from 'adm-zip';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 4000;
const MONGO_URI = process.env.MONGO_URI || 'mongodb://127.0.0.1:27017/neuro_lens';

// --- CORS Middleware (applied globally) ---
app.use(cors({
  origin: 'http://localhost:5173', // for development; replace with your frontend URL in production
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'Origin', 'Accept'],
  exposedHeaders: ['Content-Type'],
  credentials: true,
}));

// Handle preflight requests early
app.options('*', cors());

app.use(express.json());
app.use(morgan('dev'));

// --- Paths ---
const __dirnameResolved = path.resolve();
const projectRoot = path.join(__dirnameResolved, '..');
const uploadsRoot = path.join(projectRoot, 'server', 'uploads');
const sharedMlUploads = path.join(projectRoot, 'ml', 'data', 'uploads');

fs.mkdirSync(uploadsRoot, { recursive: true });
fs.mkdirSync(sharedMlUploads, { recursive: true });

// --- MongoDB Setup ---
await mongoose.connect(MONGO_URI);
const ImageSchema = new mongoose.Schema(
  {
    name: { type: String, required: true },
    url: { type: String },
    annotatedBy: { type: String },
    completed: { type: Boolean, default: false },
  },
  { timestamps: true }
);
const ImageModel = mongoose.model('Image', ImageSchema);

// --- Multer Setup ---
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadsRoot),
  filename: (req, file, cb) => {
    const ts = Date.now();
    const safe = file.originalname.replace(/[^a-zA-Z0-9_.-]/g, '_');
    cb(null, `${ts}_${safe}`);
  },
});
const upload = multer({ storage });

// --- Static Assets (Fix for OpaqueResponseBlocking) ---
app.use('/uploads', (req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*'); // Key fix
  res.setHeader('Cross-Origin-Resource-Policy', 'cross-origin'); // Important for Chrome
  next();
}, express.static(uploadsRoot));

// --- Upload Route ---
app.post('/api/datasets/upload', upload.single('dataset'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ ok: false, error: 'No file uploaded' });
    if (!req.file.originalname.toLowerCase().endsWith('.zip')) {
      return res.status(400).json({ ok: false, error: 'Only .zip accepted' });
    }

    const zipPath = req.file.path;
    const extractFolderName = path.basename(zipPath, path.extname(zipPath));
    const extractDir = path.join(uploadsRoot, extractFolderName);
    fs.mkdirSync(extractDir, { recursive: true });

    const zip = new AdmZip(zipPath);
    zip.extractAllTo(extractDir, true);

    const mirrorDir = sharedMlUploads;
    fs.mkdirSync(mirrorDir, { recursive: true });

    const validExt = new Set(['.jpg', '.jpeg', '.png', '.bmp', '.tiff']);
    const created = [];

    const walk = (dir) => {
      const entries = fs.readdirSync(dir, { withFileTypes: true });
      for (const e of entries) {
        const full = path.join(dir, e.name);
        if (e.isDirectory()) walk(full);
        else {
          const ext = path.extname(e.name).toLowerCase();
          if (validExt.has(ext)) {
            const rel = path.relative(uploadsRoot, full).split(path.sep).join('/');
            const url = `${req.protocol}://${req.get('host')}/uploads/${rel}`;
            created.push({ name: e.name, url });

            const dest = path.join(mirrorDir, e.name);
            try { fs.copyFileSync(full, dest); } catch {}
          }
        }
      }
    };
    walk(extractDir);

    if (created.length) await ImageModel.insertMany(created);

    return res.json({ ok: true, count: created.length });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ ok: false, error: 'Upload failed' });
  }
});

// --- Get Images ---
app.get('/api/images', async (req, res) => {
  const images = await ImageModel.find().sort({ createdAt: -1 }).lean();
  return res.json({ images });
});

// --- Delete All Datasets ---
app.delete('/api/datasets', async (req, res) => {
  try {
    await ImageModel.deleteMany({});

    const emptyDir = (dir) => {
      if (!fs.existsSync(dir)) return;
      fs.readdirSync(dir, { withFileTypes: true }).forEach((e) => {
        const p = path.join(dir, e.name);
        try {
          if (e.isDirectory()) {
            emptyDir(p);
            fs.rmSync(p, { recursive: true, force: true });
          } else fs.unlinkSync(p);
        } catch {}
      });
    };

    emptyDir(uploadsRoot);
    emptyDir(sharedMlUploads);
    fs.mkdirSync(uploadsRoot, { recursive: true });
    fs.mkdirSync(sharedMlUploads, { recursive: true });

    return res.json({ ok: true });
  } catch (err) {
    console.error('Failed to clear datasets', err);
    return res.status(500).json({ ok: false, error: 'Failed to clear datasets' });
  }
});

// --- Start Server ---
app.listen(PORT, () => {
  console.log(`✅ Node server listening on http://localhost:${PORT}`);
});
