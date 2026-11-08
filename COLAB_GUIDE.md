# تشغيل بصمة رقم في Google Colab — خطوة بخطوة

هذا دليل لعرض **مشروع بصمة رقم** من الحزمة المتاحة، لا لنسخ مثال بيانات الطلاب الخاص بالمدربة. إذا توفر الوقت، فإن رفع المشروع على Hugging Face شرط منفصل لا يغنيه رابط Colab المؤقت.

## الخلية 1 — رفع ملف ZIP من جهازك

```python
from google.colab import files
uploaded = files.upload()
```

اختر الملف `RaqamSense_Workshop_Aligned.zip` ثم أكمل.

## الخلية 2 — استخراج المشروع

```python
import zipfile
from pathlib import Path
name = "RaqamSense_Workshop_Aligned.zip"
project = Path('/content/RaqamSense')
project.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(name) as z:
    z.extractall(project)
print(sorted(p.name for p in project.iterdir()))
```

## الخلية 3 — تثبيت المكتبات

```python
%cd /content/RaqamSense
!pip install -q -r requirements.txt
```

## الخلية 4 — إعادة التدريب وحساب الأدلة

```python
!python train_model.py
```

تظهر الدقة المحسوبة من صور الاختبار المستقلة والعمق المختار من بيانات التحقق. راجع `reports/metrics.json` إذا احتجت التفاصيل.

## الخلية 5 — تشغيل واجهة Gradio برابط مؤقت

```python
import gradio as gr
import app
app.demo.launch(
    share=True,
    css=app.CSS,
    theme=gr.themes.Soft(
        primary_hue='violet', secondary_hue='teal', neutral_hue='slate',
        font=['Tajawal', 'system-ui', 'sans-serif']
    )
)
```

سيظهر رابط `gradio.live` مؤقت طالما بقيت الجلسة تعمل. لا تسلّمه بدل رابط Hugging Face إذا طُلِب التطبيق المنشور.

## الخلية 6 — حفظ التغييرات إن احتجت

```python
import shutil
shutil.make_archive('/content/RaqamSense_output', 'zip', '/content/RaqamSense')
files.download('/content/RaqamSense_output.zip')
```

**لجنة التقييم تطلب رابطين قبل العرض:** تطبيق منشور على Hugging Face ومستودع GitHub، كما في عرض اليوم الخامس.
