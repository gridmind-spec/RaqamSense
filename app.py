"""RaqamSense — Arabic RTL, accessible light/dark Gradio app.

Model/data source: scikit-learn's 8x8 handwritten digits.
No unsupported claims of calibrated probability are made.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
import gradio as gr
import joblib
from sklearn import __version__ as sklearn_version
from train_model import train_and_export, MODEL_PATH, METRICS_PATH, ROOT

SAMPLE_DIR = ROOT / 'app_assets' / 'samples'

def get_artifacts():
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        return train_and_export()
    try:
        pkg = joblib.load(MODEL_PATH)  # Only read locally bundled, trusted project artifact.
        if pkg.get('sklearn_version') != sklearn_version:
            return train_and_export()
        return pkg['model'], json.loads(METRICS_PATH.read_text(encoding='utf-8'))
    except (ValueError, OSError, KeyError, AttributeError):
        return train_and_export()

MODEL, METRICS = get_artifacts()


def prepare_input(image):
    """Support dataset examples and user uploads with explicit distribution warnings."""
    if image is None:
        raise gr.Error('ارفع صورة رقم، أو اختر مثالًا من الصور الجاهزة.')
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    # Flatten transparency on black; the model's training images are bright strokes on dark canvas.
    if image.mode == 'RGBA':
        flattened = Image.new('RGBA', image.size, (0, 0, 0, 255))
        flattened.alpha_composite(image)
        image = flattened.convert('RGB')
    gray = ImageOps.grayscale(image)
    data = np.array(gray, dtype=np.float32)
    if data.size == 0:
        raise gr.Error('الصورة المدخلة فارغة.')
    # Auto-detect white-background handwriting from corner pixels.
    edges = np.concatenate([data[:2,:].ravel(), data[-2:,:].ravel(), data[:,:2].ravel(), data[:,-2:].ravel()])
    if float(np.median(edges)) > 127:
        gray = ImageOps.invert(gray)
        data = np.asarray(gray, dtype=np.float32)
    # Dataset examples are 240x240 nearest-neighbor upscales of 8x8 training samples.
    # Resizing them directly preserves the original pixels and makes comparisons reproducible.
    if gray.width == 240 and gray.height == 240:
        small = gray.resize((8, 8), Image.Resampling.BOX)
    else:
        # General uploads: crop surrounding blank canvas and place digit inside a small frame.
        mask = data > 35
        if not mask.any():
            raise gr.Error('لم أجد رقمًا واضحًا في الصورة. استخدم خطًا أبيض على خلفية سوداء أو العكس.')
        yy, xx = np.where(mask)
        x0, x1, y0, y1 = max(0, xx.min()-2), min(gray.width, xx.max()+3), max(0, yy.min()-2), min(gray.height, yy.max()+3)
        cut = gray.crop((int(x0),int(y0),int(x1),int(y1)))
        cut.thumbnail((6, 6), Image.Resampling.LANCZOS)
        small = Image.new('L', (8,8), 0)
        small.paste(cut, ((8-cut.width)//2, (8-cut.height)//2))
    vector = np.asarray(small, dtype=np.float32) / 255.0 * 16.0
    return vector.reshape(1, -1), small.resize((240,240), Image.Resampling.NEAREST)


def predict_image(image):
    vec, processed = prepare_input(image)
    values = pd.DataFrame(vec, columns=[f'pixel_{i:02d}' for i in range(64)])
    predicted = int(MODEL.predict(values)[0])
    msg = (
       f'<div class="outcome" dir="rtl"><div class="overline">قراءة النموذج</div>'
       f'<div class="outcome-digit">{predicted}</div>'
       f'<div class="outcome-caption">الرقم الذي يتوقعه نموذج شجرة القرار</div>'
       f'<p class="outcome-footnote">هذه نتيجة تصنيف وليست ضمانًا للصحة. الصور المرسومة خارج مجموعة التدريب قد تكون أصعب.</p></div>'
    )
    return msg, processed


def predict_sketch(editor):
    if not editor or not isinstance(editor, dict):
        raise gr.Error('ارسم رقمًا أبيض على اللوحة السوداء أولًا.')
    image = editor.get('composite')
    if image is None:
        raise gr.Error('لم تصل صورة مرسومة من اللوحة.')
    result, processed = predict_image(image)
    return result, processed


TEST_ACC = METRICS['test_accuracy'] * 100
BASE_ACC = METRICS['majority_baseline_accuracy'] * 100
DEPTH = METRICS['selected_max_depth']

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;600;700;800&display=swap');
:root {
  --ra-primary:#5029b9; --ra-mint:#22bfb1; --ra-text:#24223a; --ra-muted:#615d78;
  --ra-page:#f8f7ff; --ra-card:#ffffff; --ra-border:#e9e5f4;
  --ra-hero:#3b1a9a; --ra-herotext:#ffffff; --ra-soft:#f1edff;
}
html.dark, body.dark, html[data-user-theme="dark"] {
  --ra-primary:#a990ff; --ra-mint:#48dcca; --ra-text:#f2effe; --ra-muted:#beb8d6;
  --ra-page:#111021; --ra-card:#1c1930; --ra-border:#37314f;
  --ra-hero:#24144f; --ra-herotext:#ffffff; --ra-soft:#2e254f;
}
html[data-user-theme="light"], body[data-user-theme="light"] {
  --ra-primary:#5029b9; --ra-mint:#22bfb1; --ra-text:#24223a; --ra-muted:#615d78;
  --ra-page:#f8f7ff; --ra-card:#ffffff; --ra-border:#e9e5f4;
  --ra-hero:#3b1a9a; --ra-herotext:#ffffff; --ra-soft:#f1edff;
}
body, .gradio-container { background:var(--ra-page)!important; color:var(--ra-text)!important; }
.gradio-container { max-width:1160px!important; direction:rtl!important; font-family:'Tajawal', 'IBM Plex Sans Arabic', system-ui, sans-serif!important; font-size:16px!important; line-height:1.7!important; }
.gradio-container * { font-family:'Tajawal','IBM Plex Sans Arabic',system-ui,sans-serif; }
.gradio-container input, .gradio-container textarea, .gradio-container select { direction:rtl!important; text-align:right!important; }
.gradio-container button { min-height:48px!important; font-weight:700!important; }
.gradio-container button:focus-visible, .gradio-container input:focus-visible {outline:3px solid var(--ra-mint)!important;outline-offset:3px!important;}
.ra-hero {position:relative; overflow:hidden; border-radius:22px; background:linear-gradient(119deg,var(--ra-hero),#6240c8 67%,#2bafa9);padding:32px 36px;color:var(--ra-herotext);margin:4px 0 18px;}
.ra-hero::before {content:'';position:absolute;inset:auto auto -90px -75px;width:300px;height:300px;border-radius:60px;border:2px solid rgba(255,255,255,.11);transform:rotate(27deg)}
.ra-eyebrow {font-size:13px;font-weight:800;letter-spacing:1.2px;color:#9af0e2;margin-bottom:8px}
.ra-hero h1 {font-size:clamp(28px,4vw,48px);line-height:1.17;color:white!important;margin:3px 0 10px;font-weight:800}
.ra-hero p {font-size:17px;line-height:1.8;color:#edeaff;max-width:760px;margin:0}
.ra-stats {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:16px 0}
.ra-stat {border:1px solid var(--ra-border);background:var(--ra-card);border-radius:16px;padding:18px 20px;color:var(--ra-text)}
.ra-stat .k {color:var(--ra-muted);font-size:13px;font-weight:700}.ra-stat .v {font-size:27px;font-weight:800;direction:ltr;text-align:right;line-height:1.2;margin-top:6px}.ra-stat .note{font-size:12px;color:var(--ra-muted);margin-top:5px}
.ra-intro {border:1px solid var(--ra-border);border-radius:14px;background:var(--ra-card);padding:14px 18px;color:var(--ra-text);margin:12px 0}
.ra-intro strong{color:var(--ra-primary)}
.outcome {min-height:205px;display:flex;justify-content:center;align-items:center;flex-direction:column;text-align:center;padding:24px;border:1px solid var(--ra-border);border-radius:18px;background:var(--ra-card);color:var(--ra-text)}
.outcome .overline {font-size:13px;color:var(--ra-muted);font-weight:700}
.outcome-digit{font-size:86px;line-height:1.15;color:var(--ra-primary);font-weight:800}
.outcome-caption{font-weight:800;font-size:17px}
.outcome-footnote{color:var(--ra-muted);font-size:13px;max-width:480px;margin:12px auto 0}
.ra-pill{display:inline-block;border-radius:99px;background:#ffffff20;border:1px solid #ffffff44;padding:5px 12px;font-size:13px;font-weight:700;margin-top:18px}
.gradio-container .prose h2, .gradio-container .prose h3 {color:var(--ra-text)!important}
.gradio-container .prose, .gradio-container .prose p {color:var(--ra-text)!important}
.gradio-container .tabs {direction:rtl!important}
.gradio-container .tab-nav {display:flex;flex-wrap:wrap;gap:6px}
#processed-preview img, #upload-image img {image-rendering:pixelated}
.ra-footer{color:var(--ra-muted);font-size:13px;text-align:center;margin:24px 0 8px}
@media(max-width:640px){.ra-hero{padding:24px 18px;border-radius:14px}.ra-stats{grid-template-columns:1fr}.ra-hero p{font-size:15px}.ra-stat .v{font-size:24px}}
"""

# gradio is configured to follow the user's system dark mode by default;
# this explicit selection additionally overrides it for the open session.
THEME_JS = """(choice) => {
  const html = document.documentElement;
  const body = document.body;
  const app = document.querySelector('gradio-app');
  if (choice === 'داكن') {
    html.classList.add('dark'); body.classList.add('dark'); app?.classList.add('dark');
    html.setAttribute('data-user-theme', 'dark'); body.setAttribute('data-user-theme', 'dark');
  } else if (choice === 'فاتح') {
    html.classList.remove('dark'); body.classList.remove('dark'); app?.classList.remove('dark');
    html.setAttribute('data-user-theme', 'light'); body.setAttribute('data-user-theme', 'light');
  } else {
    html.removeAttribute('data-user-theme'); body.removeAttribute('data-user-theme');
    const dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    html.classList.toggle('dark', dark);body.classList.toggle('dark', dark);app?.classList.toggle('dark', dark);
  }
  return [];
}"""

HEAD = f"""
<section class="ra-hero" dir="rtl" aria-label="مقدمة تطبيق بصمة رقم">
 <div class="ra-eyebrow">APPLIED MACHINE LEARNING · PROJECT SHOWCASE</div>
 <h1>بصمة رقم <span style="color:#70eddb">/ RaqamSense</span></h1>
 <p>كيف يتعلّم الحاسوب قراءة الأرقام المكتوبة باليد؟ جرّب تصنيف صورة رقم، ثم شاهد كيف يغيّر تنظيف البيانات وضبط إعدادات شجرة القرار جودة النتيجة.</p>
 <span class="ra-pill">Python · Pandas · Scikit-learn · Decision Tree · Gradio</span>
</section>
<div class="ra-stats" dir="rtl">
 <div class="ra-stat"><div class="k">الدقة على مجموعة اختبار مستقلة</div><div class="v">{TEST_ACC:.1f}%</div><div class="note">{METRICS['test_samples']} صورة لم تُستخدم في التدريب أو اختيار العمق</div></div>
 <div class="ra-stat"><div class="k">إجمالي الصور الأصلية</div><div class="v">1797</div><div class="note">أرقام حقيقية ضمن مجموعة بيانات sklearn Digits</div></div>
 <div class="ra-stat"><div class="k">عمق الشجرة المختار</div><div class="v">{DEPTH if DEPTH is not None else 'بدون حد'}</div><div class="note">اختير باستخدام التحقق Validation وليس الاختبار النهائي</div></div>
</div>
"""

HOW = f"""
<div dir="rtl" class="ra-intro"><strong>المشكلة:</strong> يمكننا تعليم نموذج تصنيف التعرّف على رقم مكتوب يدويًا من صور منخفضة الدقة، بدل كتابة قواعد يدوية لكل شكل ممكن.</div>

### دورة حياة البيانات والنموذج

1. **مصدر البيانات:** `load_digits()` من مكتبة scikit-learn. كل صورة رمادية بحجم `8×8`، أي **64 خاصية رقمية** وقيمة هدف من 0 إلى 9.
2. **التحقق والتنظيف:** فحص القيم المفقودة والمكررة، والتأكد أن البكسلات ضمن المدى `0–16`، ثم حفظ `cleaned_data.csv`.
3. **التقسيم الصحيح:** مجموعة تطوير 80% ومجموعة اختبار مستقلة 20%. نقسم التطوير إلى تدريب وتحقق لاختيار إعدادات الشجرة.
4. **التدريب:** `DecisionTreeClassifier` مع تجربة أعماق `3, 5, 7, 9, 12, None` وضبط `min_samples_leaf=2`.
5. **الاختيار والتقييم:** نختار أفضل عمق وفق `validation accuracy`، ثم نعيد التدريب على مجموعة التطوير، ونقيس الأداء مرة واحدة على الاختبار النهائي.
6. **النشر:** نموذج محفوظ وواجهة Gradio قابلة للنشر عبر Hugging Face Spaces؛ توثيق وملفات التكرار داخل GitHub.

### القراءة الصحيحة للنتائج

- **دقة الاختبار المقاسة:** **{TEST_ACC:.1f}%**، مقارنة بخط أساس تقريبي **{BASE_ACC:.1f}%** عند توقّع الفئة الأكثر تكرارًا دائمًا.
- هذا ليس ضمانًا لدقة مماثلة على صور الهاتف أو الرسومات الجديدة؛ فهي قد تختلف عن توزيع بيانات التدريب.
- لم نعرض نسبة «ثقة» مصطنعة؛ تقديرات الأوراق النهائية لشجرة القرار ليست احتمالات معايرة بالضرورة.
"""

with gr.Blocks(title='بصمة رقم | RaqamSense') as demo:
    gr.HTML(HEAD)
    with gr.Row():
        gr.Markdown('**اختر التجربة المناسبة:** أمثلة حقيقية من الاختبار أو رفع صورة جديدة.')
        theme_picker = gr.Dropdown(['تلقائي','فاتح','داكن'], value='تلقائي', label='المظهر', scale=1, min_width=155, interactive=True)
    theme_picker.change(fn=None, inputs=theme_picker, outputs=[], js=THEME_JS, queue=False)

    with gr.Tabs():
        with gr.Tab('جرّب التعرّف على رقم'):
            gr.HTML('<div class="ra-intro" dir="rtl"><strong>أفضل تجربة للعرض:</strong> اضغط إحدى صور الاختبار العشر أسفل المدخل، ثم اضغط «حلّل الرقم». تستطيع أيضًا رفع صورة تحتوي على رقم واحد فقط. يعرض النموذج الرقم المتوقع ونسخة الصورة بعد تحويلها إلى 8×8.</div>')
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    upload = gr.Image(type='pil', image_mode='RGB', sources=['upload','clipboard'], label='صورة رقم مكتوب بخط اليد', height=285, elem_id='upload-image')
                    run = gr.Button('حلّل الرقم', variant='primary', size='lg')
                with gr.Column(scale=1):
                    result = gr.HTML('<div class="outcome" dir="rtl"><span class="overline">النتيجة</span><span class="outcome-caption">أدخل صورة لبدء التجربة</span></div>')
                    processed = gr.Image(label='ما يراه النموذج بعد المعالجة (8×8 مكبرة)', interactive=False, height=180, elem_id='processed-preview', image_mode='L')
            run.click(predict_image, inputs=upload, outputs=[result,processed], api_name='classify_digit')
            files = [str(SAMPLE_DIR / f'digit_{i}.png') for i in range(10)]
            gr.Examples(examples=[[file] for file in files], inputs=upload,
                        label='أمثلة قابلة للنقر من مجموعة الاختبار (0–9)', example_labels=[f'الرقم {i}' for i in range(10)])

        with gr.Tab('ارسم رقمًا بنفسك'):
            gr.HTML('<div class="ra-intro" dir="rtl"><strong>تجربة تحدٍ:</strong> ارسم خطًا أبيض سميكًا على الخلفية السوداء. هذه التجربة أصعب للنموذج لأن خطوط الرسم الحر قد لا تشبه بياناته التدريبية؛ الخطأ هنا فرصة لشرح مفهوم <b>توزيع البيانات</b> لا خللًا يجب إخفاؤه.</div>')
            with gr.Row():
                with gr.Column(scale=1):
                    sketch = gr.ImageEditor(
                        value=Image.new('RGB',(256,256),'black'), type='pil', image_mode='RGB',
                        canvas_size=(256,256), fixed_canvas=True, transforms=[], layers=False,
                        brush=gr.Brush(default_size=22,colors=['#ffffff'],default_color='#ffffff',color_mode='fixed'),
                        eraser=gr.Eraser(default_size=22), label='لوحة رسم الرقم', height=360)
                    draw_run = gr.Button('تعرّف على الرسم', variant='primary')
                with gr.Column(scale=1):
                    draw_result = gr.HTML('<div class="outcome" dir="rtl">النتيجة ستظهر هنا بعد الرسم.</div>')
                    draw_processed = gr.Image(label='الصورة التي دخلت للنموذج', height=190, interactive=False, elem_id='processed-preview')
            draw_run.click(predict_sketch, inputs=sketch, outputs=[draw_result,draw_processed], api_name='classify_sketch')

        with gr.Tab('المنهجية والدليل'):
            gr.Markdown(HOW, elem_id='methodology')
            with gr.Accordion('الإعدادات التي جربناها بالفعل', open=False):
                gr.Dataframe(value=pd.read_csv(ROOT/'reports'/'hyperparameter_search.csv'), interactive=False, label='Accuracy على Validation (وليس Test)')
            with gr.Accordion('مصفوفة الأخطاء للفئات العشر', open=False):
                gr.Dataframe(value=pd.read_csv(ROOT/'reports'/'confusion_matrix.csv'),interactive=False,label='الصف: الحقيقة · العمود: التوقع')

        with gr.Tab('عن المشروع'):
            gr.Markdown('''### لماذا هذا المشروع؟

تطبيق مباشر للمهارات التي شرحتها الورشة: Data Cleaning، Feature Engineering، Supervised Classification، Hyperparameter Tuning، Gradio، وتوثيق GitHub. استخدمنا بيانات معروفة وقابلة لإعادة الإنتاج بدل بيانات غير قابلة للتحقق.

**حدود النظام:** نموذج تعليمي ليس خدمة رسمية لقراءة الوثائق، ولا يجب استخدامه في قرارات عالية الأثر. الدقة المذكورة تخص مجموعة الاختبار المحددة فقط.

**مصدر البيانات:** [Scikit-learn Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).

**الهوية البصرية:** مستوحاة من البنفسجي والتركواز الواردين في عروض الدورة، مع دعم RTL والمظهر الفاتح والداكن، دون استخدام شعارات رسمية أو الادعاء بملكية الجهة المنظمة.

**الترخيص:** MIT للمشروع البرمجي. توثيق البيانات في README.
''')
    gr.HTML('<div class="ra-footer" dir="rtl">RaqamSense · مشروع تعليمي في تعلم الآلة التطبيقي · نتائج قابلة لإعادة الإنتاج</div>')

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860, show_error=True, theme=gr.themes.Soft(primary_hue='violet', secondary_hue='teal', neutral_hue='slate', font=['Tajawal','system-ui','sans-serif']), css=CSS)
