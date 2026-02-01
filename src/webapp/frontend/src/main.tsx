import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

// #region agent log
fetch('http://127.0.0.1:7243/ingest/fc71f522-f272-4d23-8f35-daabdd192e3f',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.tsx',message:'main.tsx executing after imports',data:{},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'H2'})}).catch(()=>{});
// #endregion

const rootEl = document.getElementById('root')
// #region agent log
fetch('http://127.0.0.1:7243/ingest/fc71f522-f272-4d23-8f35-daabdd192e3f',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.tsx',message:'before createRoot',data:{rootExists:!!rootEl},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'H3'})}).catch(()=>{});
// #endregion

try {
  createRoot(rootEl!).render(
    <StrictMode>
      <App />
    </StrictMode>,
  )
  // #region agent log
  fetch('http://127.0.0.1:7243/ingest/fc71f522-f272-4d23-8f35-daabdd192e3f',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.tsx',message:'createRoot.render completed',data:{},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'H4'})}).catch(()=>{});
  // #endregion
} catch (e) {
  // #region agent log
  fetch('http://127.0.0.1:7243/ingest/fc71f522-f272-4d23-8f35-daabdd192e3f',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.tsx',message:'createRoot/render threw',data:{error:String(e)},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'H5'})}).catch(()=>{});
  // #endregion
  throw e
}
