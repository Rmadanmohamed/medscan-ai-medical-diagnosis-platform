import streamlit as st


DISCLAIMER = (
    "This project is for educational and portfolio demonstration only. "
    "It is not a medical diagnosis tool and should not be used for clinical decisions."
)


def show_medical_image_demo() -> None:
    st.subheader("Medical Image Demo")
    uploaded_file = st.file_uploader(
        "Upload a sample image",
        type=["jpg", "jpeg", "png"],
        help="Use a non-sensitive sample image. No clinical diagnosis is produced.",
    )

    if uploaded_file is None:
        st.info("Upload a sample image to preview the demo workflow.")
        return

    st.image(uploaded_file, caption="Uploaded sample image", use_container_width=True)
    st.warning(
        "Demo mode: no validated trained model is bundled with this repository, "
        "so the app does not generate a medical prediction."
    )


def show_retinal_demo() -> None:
    st.subheader("Retinal Image Demo")
    uploaded_file = st.file_uploader(
        "Upload a sample retinal image",
        type=["jpg", "jpeg", "png"],
        key="retinal_upload",
        help="Use a non-sensitive sample image. No clinical diagnosis is produced.",
    )

    if uploaded_file is None:
        st.info("Upload a sample image to preview the retinal workflow.")
        return

    st.image(uploaded_file, caption="Uploaded sample retinal image", use_container_width=True)
    st.warning(
        "Demo mode: retinal analysis requires a validated model and dataset before any "
        "medical-risk output can be presented."
    )


def show_risk_demo() -> None:
    st.subheader("Tabular Risk-Screening Demo")
    age = st.number_input("Age", min_value=1, max_value=120, value=35)
    bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=24.5, step=0.1)
    systolic_bp = st.number_input("Systolic blood pressure", min_value=70, max_value=250, value=120)
    glucose = st.number_input("Glucose", min_value=40, max_value=400, value=95)

    if st.button("Run demo screening"):
        demo_score = min(1.0, max(0.0, (age / 120 * 0.25) + (bmi / 70 * 0.25) + (systolic_bp / 250 * 0.25) + (glucose / 400 * 0.25)))
        st.metric("Demo score", f"{demo_score:.0%}")
        st.caption(
            "This score is a transparent interface placeholder, not a trained medical model output."
        )


def main() -> None:
    st.set_page_config(page_title="MedScan AI Demo", layout="wide")
    st.title("MedScan AI Medical Demo")
    st.info(DISCLAIMER)

    st.markdown(
        "This safe portfolio entrypoint demonstrates the user experience without "
        "claiming clinical accuracy or producing diagnoses."
    )

    tab_image, tab_retinal, tab_risk = st.tabs(
        ["Medical image", "Retinal image", "Risk screening"]
    )

    with tab_image:
        show_medical_image_demo()

    with tab_retinal:
        show_retinal_demo()

    with tab_risk:
        show_risk_demo()

    st.divider()
    st.caption("Author: Ramadan Mohamed Hassan")


if __name__ == "__main__":
    main()
