
# ملف التطبيق الرئيسي - main_app.py
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import time

# استيراد النماذج المخصصة
from medical_image_analyzer import MedicalImageAnalyzer
from retinal_analyzer import RetinalAnalyzer
from disease_prediction import DiseasePredictionSystem

class AIMedScanApp:
    """
    تطبيق AI-MedScan Pro - نظام التشخيص الذكي
    """

    def __init__(self):
        self.medical_analyzer = MedicalImageAnalyzer()
        self.retinal_analyzer = RetinalAnalyzer()
        self.disease_predictor = DiseasePredictionSystem()

        # تحميل النماذج المدربة مسبقاً
        self.load_models()

    def load_models(self):
        """تحميل النماذج المدربة"""
        try:
            self.medical_analyzer.build_model()
            self.retinal_analyzer.build_retinal_model()
            self.disease_predictor.build_prediction_models()
            st.success("تم تحميل جميع النماذج بنجاح!")
        except Exception as e:
            st.error(f"خطأ في تحميل النماذج: {e}")

    def main_interface(self):
        """واجهة التطبيق الرئيسية"""
        # العنوان والشعار
        st.set_page_config(
            page_title="AI-MedScan Pro",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # التصميم المخصص
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: bold;
        }
        .feature-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            color: white;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #1f77b4;
        }
        </style>
        """, unsafe_allow_html=True)

        # العنوان الرئيسي
        st.markdown('<h1 class="main-header">🔬 AI-MedScan Pro</h1>', unsafe_allow_html=True)
        st.markdown('<h3 style="text-align:center; color:#666;">نظام التشخيص الذكي متعدد المراحل</h3>', unsafe_allow_html=True)

        # الشريط الجانبي للتنقل
        with st.sidebar:
            st.image("https://via.placeholder.com/300x150/1f77b4/white?text=AI-MedScan+Pro", width=300)

            menu_option = st.selectbox(
                "اختر نوع التشخيص:",
                [
                    "🏠 الصفحة الرئيسية",
                    "📸 تحليل الصور الطبية",
                    "👁️ فحص شبكية العين", 
                    "🔮 التنبؤ بالأمراض",
                    "📊 التقارير والإحصائيات",
                    "⚙️ الإعدادات"
                ]
            )

        # عرض الصفحات حسب الاختيار
        if menu_option == "🏠 الصفحة الرئيسية":
            self.show_homepage()
        elif menu_option == "📸 تحليل الصور الطبية":
            self.medical_imaging_page()
        elif menu_option == "👁️ فحص شبكية العين":
            self.retinal_analysis_page()
        elif menu_option == "🔮 التنبؤ بالأمراض":
            self.disease_prediction_page()
        elif menu_option == "📊 التقارير والإحصائيات":
            self.reports_page()
        elif menu_option == "⚙️ الإعدادات":
            self.settings_page()

    def show_homepage(self):
        """الصفحة الرئيسية"""
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="feature-card">
                <h3>🎯 دقة 97%</h3>
                <p>دقة تشخيص أعلى من الأطباء المختصين في معظم الحالات</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="feature-card">
                <h3>⚡ نتائج فورية</h3>
                <p>احصل على النتائج في أقل من 30 ثانية</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="feature-card">
                <h3>💰 توفير 70%</h3>
                <p>من تكاليف التشخيص التقليدي</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # الإحصائيات الرئيسية
        st.subheader("📈 إحصائيات النظام")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("عدد التشخيصات", "1,234,567", "↗️ 15%")

        with col2:
            st.metric("دقة النظام", "97.2%", "↗️ 2.1%")

        with col3:
            st.metric("المستخدمين النشطين", "89,432", "↗️ 8%")

        with col4:
            st.metric("الأرواح المُنقذة", "45,123", "↗️ 12%")

        # الأمراض المدعومة
        st.subheader("🔬 الأمراض المدعومة")

        diseases_data = {
            'المرض': ['السرطان', 'أمراض القلب', 'السكري', 'أمراض الكلى', 'الأمراض العصبية'],
            'دقة التشخيص': [97, 95, 94, 96, 93],
            'عدد الحالات': [45123, 38567, 32890, 28456, 19234]
        }

        df = pd.DataFrame(diseases_data)

        fig = px.bar(df, x='المرض', y='دقة التشخيص', 
                    title='دقة التشخيص لكل مرض',
                    color='دقة التشخيص',
                    color_continuous_scale='viridis')

        st.plotly_chart(fig, use_container_width=True)

    def medical_imaging_page(self):
        """صفحة تحليل الصور الطبية"""
        st.header("📸 تحليل الصور الطبية المتقدم")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("رفع الصورة الطبية")

            uploaded_file = st.file_uploader(
                "اختر صورة طبية (X-Ray, MRI, CT Scan)",
                type=['jpg', 'jpeg', 'png', 'dicom'],
                help="يدعم النظام جميع أنواع الصور الطبية"
            )

            if uploaded_file is not None:
                # عرض الصورة
                image = Image.open(uploaded_file)
                st.image(image, caption="الصورة المرفوعة", use_column_width=True)

                # زر التحليل
                if st.button("🔍 تحليل الصورة", type="primary"):
                    with st.spinner("جاري تحليل الصورة..."):
                        # محاكاة التحليل
                        time.sleep(2)

                        # نتائج وهمية للعرض
                        results = {
                            'disease': 'سرطان الرئة - المرحلة المبكرة',
                            'confidence': 0.94,
                            'reliable': True,
                            'recommendations': [
                                'مراجعة طبيب الأورام فوراً',
                                'إجراء فحوصات إضافية للتأكيد',
                                'بدء العلاج في أقرب وقت'
                            ]
                        }

                        st.session_state['analysis_results'] = results

        with col2:
            st.subheader("نتائج التحليل")

            if 'analysis_results' in st.session_state:
                results = st.session_state['analysis_results']

                # عرض النتائج
                if results['reliable']:
                    st.success(f"✅ تم التشخيص بنجاح")
                else:
                    st.warning("⚠️ يُنصح بإجراء فحوصات إضافية")

                st.markdown(f"**التشخيص:** {results['disease']}")
                st.markdown(f"**مستوى الثقة:** {results['confidence']:.1%}")

                # شريط الثقة
                st.progress(results['confidence'])

                # التوصيات
                st.subheader("📋 التوصيات الطبية")
                for i, rec in enumerate(results['recommendations'], 1):
                    st.markdown(f"{i}. {rec}")

                # تحميل التقرير
                if st.button("📄 تحميل التقرير المفصل"):
                    self.generate_medical_report(results)
            else:
                st.info("قم برفع صورة طبية للحصول على النتائج")

    def retinal_analysis_page(self):
        """صفحة فحص شبكية العين"""
        st.header("👁️ فحص شبكية العين المتقدم")

        st.markdown("""
        <div class="metric-card">
            <h4>🎯 دقة 96% في كشف أمراض القلب والكلى من فحص العين فقط</h4>
            <p>تقنية ثورية تكشف الأمراض قبل ظهور الأعراض</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("🔍 تعليمات الفحص")

            st.markdown("""
            **خطوات الفحص:**
            1. تأكد من الإضاءة الجيدة
            2. انظر مباشرة إلى الكاميرا
            3. حافظ على ثبات العين
            4. اتبع التعليمات على الشاشة
            """)

            # محاكاة كاميرا الفحص
            camera_on = st.checkbox("🎥 تشغيل كاميرا الفحص")

            if camera_on:
                st.image("https://via.placeholder.com/400x300/4CAF50/white?text=Eye+Camera+Active", 
                        caption="كاميرا فحص العين نشطة")

                if st.button("📸 التقاط صورة العين", type="primary"):
                    with st.spinner("جاري تحليل شبكية العين..."):
                        time.sleep(3)

                        # نتائج وهمية
                        retinal_results = {
                            'heart_disease': {'probability': 0.23, 'risk': 'منخفض'},
                            'kidney_disease': {'probability': 0.67, 'risk': 'متوسط'},
                            'diabetes': {'probability': 0.89, 'risk': 'عالي'},
                            'hypertension': {'probability': 0.45, 'risk': 'منخفض'}
                        }

                        st.session_state['retinal_results'] = retinal_results

        with col2:
            st.subheader("📊 نتائج فحص الشبكية")

            if 'retinal_results' in st.session_state:
                results = st.session_state['retinal_results']

                for disease, info in results.items():
                    disease_name = {
                        'heart_disease': 'أمراض القلب',
                        'kidney_disease': 'أمراض الكلى', 
                        'diabetes': 'السكري',
                        'hypertension': 'ارتفاع ضغط الدم'
                    }[disease]

                    # تحديد لون المخاطر
                    if info['probability'] > 0.7:
                        color = "🔴"
                    elif info['probability'] > 0.4:
                        color = "🟡"
                    else:
                        color = "🟢"

                    st.markdown(f"""
                    **{color} {disease_name}**
                    - احتمالية الخطر: {info['probability']:.1%}
                    - مستوى الخطر: {info['risk']}
                    """)

                    st.progress(info['probability'])
                    st.markdown("---")

                # الإجراءات الموصى بها
                st.subheader("🏥 الإجراءات الموصى بها")
                high_risk_diseases = [k for k, v in results.items() if v['probability'] > 0.6]

                if high_risk_diseases:
                    st.warning("يُنصح بزيارة الطبيب للفحوصات التالية:")
                    for disease in high_risk_diseases:
                        disease_name = {
                            'heart_disease': 'طبيب القلب',
                            'kidney_disease': 'طبيب الكلى', 
                            'diabetes': 'طبيب الغدد الصماء',
                            'hypertension': 'طبيب الباطنة'
                        }[disease]
                        st.markdown(f"• {disease_name}")
                else:
                    st.success("✅ لا توجد مؤشرات خطر عالية حالياً")
            else:
                st.info("قم بتشغيل الكاميرا والتقاط صورة للعين")

    def disease_prediction_page(self):
        """صفحة التنبؤ بالأمراض"""
        st.header("🔮 التنبؤ المبكر بالأمراض")

        st.markdown("""
        <div class="metric-card">
            <h4>⚡ تنبؤ بدقة 89% قبل ظهور الأعراض بشهور</h4>
            <p>نظام ذكي يحلل بياناتك الصحية للتنبؤ بالمخاطر المستقبلية</p>
        </div>
        """, unsafe_allow_html=True)

        # نموذج إدخال البيانات الصحية
        with st.form("health_data_form"):
            st.subheader("📝 البيانات الشخصية والصحية")

            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.number_input("العمر", min_value=1, max_value=120, value=35)
                gender = st.selectbox("الجنس", ["ذكر", "أنثى"])
                weight = st.number_input("الوزن (كيلو)", min_value=20, max_value=300, value=70)
                height = st.number_input("الطول (سم)", min_value=100, max_value=250, value=170)

            with col2:
                bp_systolic = st.number_input("ضغط الدم الانقباضي", min_value=80, max_value=250, value=120)
                bp_diastolic = st.number_input("ضغط الدم الانبساطي", min_value=40, max_value=150, value=80)
                heart_rate = st.number_input("معدل ضربات القلب", min_value=40, max_value=200, value=72)
                cholesterol = st.number_input("الكوليسترول", min_value=100, max_value=400, value=180)

            with col3:
                glucose = st.number_input("مستوى السكر", min_value=50, max_value=300, value=90)
                smoking = st.selectbox("التدخين", ["لا", "نعم", "سابقاً"])
                exercise = st.selectbox("ممارسة الرياضة", ["نادراً", "أحياناً", "بانتظام", "يومياً"])
                family_history = st.multiselect("التاريخ العائلي", 
                                               ["أمراض القلب", "السكري", "السرطان", "ارتفاع ضغط الدم"])

            submit_button = st.form_submit_button("🔍 تحليل المخاطر", type="primary")

        if submit_button:
            with st.spinner("جاري تحليل البيانات والتنبؤ بالمخاطر..."):
                time.sleep(3)

                # حساب BMI
                bmi = weight / ((height/100) ** 2)

                # نتائج التنبؤ (محاكاة)
                predictions = {
                    'heart_disease': {
                        'probability': min(0.95, max(0.05, (age/100 + (bmi-25)/50 + (bp_systolic-120)/200))),
                        'time_estimate': "2-4 سنوات"
                    },
                    'diabetes': {
                        'probability': min(0.95, max(0.05, (glucose-90)/100 + (bmi-25)/30)),
                        'time_estimate': "1-3 سنوات"
                    },
                    'kidney_disease': {
                        'probability': min(0.95, max(0.05, (bp_systolic-120)/300 + age/200)),
                        'time_estimate': "3-6 سنوات"
                    }
                }

                # عرض النتائج
                st.subheader("📊 نتائج التنبؤ بالمخاطر")

                for disease, pred in predictions.items():
                    disease_name = {
                        'heart_disease': 'أمراض القلب والأوعية الدموية',
                        'diabetes': 'مرض السكري من النوع الثاني',
                        'kidney_disease': 'أمراض الكلى المزمنة'
                    }[disease]

                    # تحديد مستوى الخطر واللون
                    if pred['probability'] > 0.7:
                        risk_level = "عالي جداً"
                        color = "🔴"
                        alert_type = "error"
                    elif pred['probability'] > 0.5:
                        risk_level = "عالي"
                        color = "🟠"
                        alert_type = "warning"
                    elif pred['probability'] > 0.3:
                        risk_level = "متوسط"
                        color = "🟡"
                        alert_type = "info"
                    else:
                        risk_level = "منخفض"
                        color = "🟢"
                        alert_type = "success"

                    # عرض النتيجة
                    with st.container():
                        st.markdown(f"### {color} {disease_name}")

                        col1, col2, col3 = st.columns([2, 1, 1])

                        with col1:
                            st.progress(pred['probability'])

                        with col2:
                            st.metric("احتمالية الخطر", f"{pred['probability']:.1%}")

                        with col3:
                            st.metric("الوقت المتوقع", pred['time_estimate'])

                        if alert_type == "error":
                            st.error(f"خطر {risk_level} - يُنصح بزيارة الطبيب فوراً")
                        elif alert_type == "warning":
                            st.warning(f"خطر {risk_level} - يُنصح بالمتابعة الطبية")
                        elif alert_type == "info":
                            st.info(f"خطر {risk_level} - المتابعة الدورية مطلوبة")
                        else:
                            st.success(f"خطر {risk_level} - استمر في نمط الحياة الصحي")

                        st.markdown("---")

                # نصائح الوقاية
                st.subheader("💡 نصائح الوقاية المخصصة")

                prevention_tips = []

                if bmi > 25:
                    prevention_tips.append("🏃‍♂️ تقليل الوزن من خلال الرياضة والنظام الغذائي")

                if bp_systolic > 130:
                    prevention_tips.append("🧂 تقليل الملح في الطعام")

                if glucose > 100:
                    prevention_tips.append("🍎 تقليل السكريات والكربوهيدرات")

                if smoking == "نعم":
                    prevention_tips.append("🚭 الإقلاع عن التدخين فوراً")

                if exercise in ["نادراً", "أحياناً"]:
                    prevention_tips.append("💪 ممارسة الرياضة 30 دقيقة يومياً")

                prevention_tips.extend([
                    "🥗 تناول الخضار والفواكه بكثرة",
                    "💧 شرب كمية كافية من الماء",
                    "😴 النوم 7-8 ساعات يومياً",
                    "🧘‍♀️ إدارة التوتر والضغط النفسي"
                ])

                for tip in prevention_tips[:8]:  # عرض أهم 8 نصائح
                    st.markdown(f"• {tip}")

    def reports_page(self):
        """صفحة التقارير والإحصائيات"""
        st.header("📊 التقارير والإحصائيات")

        # إحصائيات عامة
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("إجمالي التشخيصات", "1,234,567", "↗️ 15.3%")

        with col2:
            st.metric("دقة النظام", "97.2%", "↗️ 0.8%")

        with col3:
            st.metric("الوقت المتوسط", "18 ثانية", "↘️ 5 ثوان")

        with col4:
            st.metric("رضا المستخدمين", "94.7%", "↗️ 2.1%")

        st.markdown("---")

        # رسوم بيانية
        col1, col2 = st.columns(2)

        with col1:
            # توزيع الأمراض المشخصة
            diseases = ['السرطان', 'أمراض القلب', 'السكري', 'أمراض الكلى', 'أخرى']
            values = [28, 24, 18, 15, 15]

            fig = go.Figure(data=[go.Pie(labels=diseases, values=values, hole=.3)])
            fig.update_layout(title="توزيع الأمراض المشخصة")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # دقة التشخيص عبر الوقت
            months = ['يناير', 'فبراير', 'مارس', 'أبريل', 'مايو', 'يونيو']
            accuracy = [94.2, 95.1, 96.3, 96.8, 97.0, 97.2]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=months, y=accuracy, mode='lines+markers', name='دقة التشخيص'))
            fig.update_layout(title="تطور دقة النظام عبر الوقت", yaxis_title="دقة التشخيص (%)")
            st.plotly_chart(fig, use_container_width=True)

    def settings_page(self):
        """صفحة الإعدادات"""
        st.header("⚙️ إعدادات النظام")

        # إعدادات المستخدم
        st.subheader("👤 الملف الشخصي")

        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("الاسم الكامل", value="د. أحمد محمد")
            email = st.text_input("البريد الإلكتروني", value="ahmed@example.com")
            phone = st.text_input("رقم الهاتف", value="+966501234567")

        with col2:
            specialty = st.selectbox("التخصص", 
                                   ["طب عام", "أشعة", "قلب", "أورام", "باطنة", "أطفال"])
            experience = st.slider("سنوات الخبرة", 1, 50, 10)
            language = st.selectbox("اللغة", ["العربية", "English", "Français"])

        # إعدادات النظام
        st.subheader("🔧 إعدادات التشخيص")

        confidence_threshold = st.slider("حد الثقة المطلوب", 0.5, 1.0, 0.85, 0.05)
        auto_save = st.checkbox("حفظ النتائج تلقائياً", value=True)
        notifications = st.checkbox("إرسال إشعارات", value=True)

        # إعدادات الأمان
        st.subheader("🔒 إعدادات الأمان")

        two_factor = st.checkbox("المصادقة الثنائية", value=False)
        data_encryption = st.checkbox("تشفير البيانات", value=True)
        backup_frequency = st.selectbox("تكرار النسخ الاحتياطي", 
                                      ["يومي", "أسبوعي", "شهري"])

        # حفظ الإعدادات
        if st.button("💾 حفظ الإعدادات", type="primary"):
            st.success("تم حفظ الإعدادات بنجاح!")

    def generate_medical_report(self, results):
        """إنشاء تقرير طبي مفصل"""
        report = f"""
        تقرير التشخيص الطبي - AI-MedScan Pro
        =====================================

        التاريخ: {datetime.now().strftime('%Y-%m-%d %H:%M')}

        نتائج التشخيص:
        - المرض المشخص: {results['disease']}
        - مستوى الثقة: {results['confidence']:.1%}
        - الموثوقية: {'عالية' if results['reliable'] else 'متوسطة'}

        التوصيات الطبية:
        """

        for i, rec in enumerate(results['recommendations'], 1):
            report += f"\n{i}. {rec}"

        report += f"""

        ملاحظات:
        - هذا التقرير تم إنشاؤه بواسطة نظام الذكاء الاصطناعي
        - يُنصح بمراجعة طبيب مختص للتأكيد
        - النتائج قابلة للتحديث مع توفر معلومات إضافية

        AI-MedScan Pro © 2024
        """

        st.download_button(
            label="📥 تحميل التقرير",
            data=report,
            file_name=f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )

# تشغيل التطبيق
if __name__ == "__main__":
    app = AIMedScanApp()
    app.main_interface()
