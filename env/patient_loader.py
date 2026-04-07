"""
AEVIS — Simulated patient cases representing retinal scan findings.
In production, these come from the AEVIS fundus camera + APTOS 2019 dataset.
For OpenEnv evaluation, we use structured text descriptions of retinal features.
"""

PATIENT_CASES = [
    # ── NORMAL cases ──────────────────────────────────────────────
    {
        "patient_id": "P001",
        "patient_age": 34,
        "diabetes_years": None,
        "ground_truth_label": "normal",
        "ground_truth_urgency": "monitor",
        "image_description": (
            "Clear retinal image. Optic disc appears healthy with sharp margins. "
            "Cup-to-disc ratio 0.3. Macula is flat and even. Blood vessels show "
            "normal calibre and branching pattern. No haemorrhages, exudates, "
            "or neovascularisation observed. Background retina appears uniform."
        ),
    },
    {
        "patient_id": "P002",
        "patient_age": 28,
        "diabetes_years": None,
        "ground_truth_label": "normal",
        "ground_truth_urgency": "monitor",
        "image_description": (
            "Well-focused fundus image. Optic nerve head is pink with distinct rim. "
            "Arteries and veins in normal 2:3 ratio. Foveal reflex present. "
            "Peripheral retina unremarkable. No pathological findings."
        ),
    },
    {
        "patient_id": "P003",
        "patient_age": 45,
        "diabetes_years": 0,
        "ground_truth_label": "normal",
        "ground_truth_urgency": "monitor",
        "image_description": (
            "Retina appears healthy. Small optic cup. Vessels course normally "
            "from disc. No drusen, no pigment changes. Macula centrally located "
            "with normal foveal light reflex. No lesions noted."
        ),
    },

    # ── MILD DR cases ─────────────────────────────────────────────
    {
        "patient_id": "P004",
        "patient_age": 52,
        "diabetes_years": 6,
        "ground_truth_label": "mild_dr",
        "ground_truth_urgency": "monitor",
        "image_description": (
            "A few microaneurysms noted in the temporal quadrant, appearing as "
            "tiny red dots. No haemorrhages. No hard exudates. Optic disc looks "
            "normal. Macula unaffected. Mild background diabetic retinopathy pattern. "
            "Overall retinal vasculature shows early diabetic changes."
        ),
    },
    {
        "patient_id": "P005",
        "patient_age": 48,
        "diabetes_years": 4,
        "ground_truth_label": "mild_dr",
        "ground_truth_urgency": "monitor",
        "image_description": (
            "Scattered microaneurysms visible near arcades. One dot haemorrhage "
            "in inferior quadrant. No cotton wool spots. Hard exudates absent. "
            "Optic nerve normal. No macular oedema suspected. Early NPDR."
        ),
    },

    # ── MODERATE DR cases ─────────────────────────────────────────
    {
        "patient_id": "P006",
        "patient_age": 58,
        "diabetes_years": 10,
        "ground_truth_label": "moderate_dr",
        "ground_truth_urgency": "refer",
        "image_description": (
            "Multiple dot and blot haemorrhages in all four quadrants. Hard exudates "
            "forming a partial ring near the macula. Several microaneurysms. "
            "Cotton wool spots present (3 visible). Venous beading noted in one vessel. "
            "No neovascularisation. Moderate non-proliferative diabetic retinopathy."
        ),
    },
    {
        "patient_id": "P007",
        "patient_age": 61,
        "diabetes_years": 12,
        "ground_truth_label": "moderate_dr",
        "ground_truth_urgency": "refer",
        "image_description": (
            "Numerous haemorrhages bilaterally visible. Hard exudates in posterior pole. "
            "Intraretinal microvascular abnormalities (IRMA) suspected. No disc "
            "neovascularisation. Macula shows early oedema signs. Refer for "
            "ophthalmologist evaluation and fluorescein angiography."
        ),
    },
    {
        "patient_id": "P008",
        "patient_age": 55,
        "diabetes_years": 8,
        "ground_truth_label": "moderate_dr",
        "ground_truth_urgency": "refer",
        "image_description": (
            "Moderate retinopathy changes. Multiple flame-shaped haemorrhages. "
            "Two cotton wool spots in superior quadrant. Hard exudates around fovea. "
            "Venous dilation noted. No active neovascularisation detected yet."
        ),
    },

    # ── SEVERE DR cases ───────────────────────────────────────────
    {
        "patient_id": "P009",
        "patient_age": 64,
        "diabetes_years": 18,
        "ground_truth_label": "severe_dr",
        "ground_truth_urgency": "urgent",
        "image_description": (
            "Extensive haemorrhages in all four quadrants meeting 4-2-1 rule. "
            "Significant venous beading in two or more quadrants. IRMA in one quadrant. "
            "Multiple cotton wool spots. Hard exudates extending toward foveal centre. "
            "No neovascularisation yet, but high risk of progression. Severe NPDR."
        ),
    },
    {
        "patient_id": "P010",
        "patient_age": 69,
        "diabetes_years": 22,
        "ground_truth_label": "severe_dr",
        "ground_truth_urgency": "urgent",
        "image_description": (
            "Very severe non-proliferative changes. Dense haemorrhages throughout. "
            "Venous loops visible. Extensive IRMA. Macular oedema clinically significant. "
            "Immediate referral to ophthalmologist strongly indicated. Risk of imminent "
            "proliferative conversion."
        ),
    },

    # ── PROLIFERATIVE DR cases ────────────────────────────────────
    {
        "patient_id": "P011",
        "patient_age": 67,
        "diabetes_years": 25,
        "ground_truth_label": "proliferative_dr",
        "ground_truth_urgency": "urgent",
        "image_description": (
            "New vessels clearly visible on the optic disc (NVD) and elsewhere on retina (NVE). "
            "Pre-retinal haemorrhage noted in inferior quadrant. Fibrovascular proliferation. "
            "Tractional changes near posterior pole. Vision-threatening proliferative diabetic "
            "retinopathy requiring urgent laser or anti-VEGF treatment."
        ),
    },
    {
        "patient_id": "P012",
        "patient_age": 72,
        "diabetes_years": 30,
        "ground_truth_label": "proliferative_dr",
        "ground_truth_urgency": "urgent",
        "image_description": (
            "Advanced proliferative retinopathy. Vitreous haemorrhage partially obscuring view. "
            "Neovascularisation at disc prominent. Fibrous traction bands visible. "
            "High risk of retinal detachment. Urgent vitreoretinal surgery consultation required."
        ),
    },

    # ── GLAUCOMA SUSPECT cases ────────────────────────────────────
    {
        "patient_id": "P013",
        "patient_age": 56,
        "diabetes_years": None,
        "ground_truth_label": "glaucoma_suspect",
        "ground_truth_urgency": "refer",
        "image_description": (
            "Enlarged cup-to-disc ratio of 0.7 noted. Disc rim thinning in inferior quadrant. "
            "Flame-shaped haemorrhage at disc margin. No diabetic changes. Optic nerve fibre "
            "layer appears thinned nasally. Suspicious for glaucomatous optic neuropathy."
        ),
    },
    {
        "patient_id": "P014",
        "patient_age": 62,
        "diabetes_years": None,
        "ground_truth_label": "glaucoma_suspect",
        "ground_truth_urgency": "refer",
        "image_description": (
            "Cup-to-disc ratio 0.75 with notching of inferior rim. Asymmetric cupping "
            "compared to fellow eye. Possible RNFL defect. No haemorrhages or exudates. "
            "Recommend IOP measurement, visual field testing, and OCT of optic nerve."
        ),
    },

    # ── AMD cases ─────────────────────────────────────────────────
    {
        "patient_id": "P015",
        "patient_age": 74,
        "diabetes_years": None,
        "ground_truth_label": "amd",
        "ground_truth_urgency": "refer",
        "image_description": (
            "Multiple large drusen (>125 microns) scattered in macular region. "
            "Pigmentary changes in RPE layer. No choroidal neovascularisation visible yet. "
            "Intermediate AMD pattern. Risk of conversion to wet AMD. "
            "Recommend Amsler grid monitoring and ophthalmology follow-up."
        ),
    },
    {
        "patient_id": "P016",
        "patient_age": 79,
        "diabetes_years": None,
        "ground_truth_label": "amd",
        "ground_truth_urgency": "urgent",
        "image_description": (
            "Subretinal fluid visible in macular area suggesting choroidal neovascularisation. "
            "Subretinal haemorrhage present. Disciform scar formation early stage. "
            "Wet AMD pattern. Urgent anti-VEGF injection referral required to preserve vision."
        ),
    },

    # ── Additional mixed cases for variety ────────────────────────
    {
        "patient_id": "P017",
        "patient_age": 41,
        "diabetes_years": 2,
        "ground_truth_label": "normal",
        "ground_truth_urgency": "monitor",
        "image_description": (
            "Early diabetic patient, retina currently within normal limits. "
            "No microaneurysms detected. Disc and macula normal. Vessels unremarkable. "
            "Annual screening recommended given diabetes duration."
        ),
    },
    {
        "patient_id": "P018",
        "patient_age": 53,
        "diabetes_years": 7,
        "ground_truth_label": "mild_dr",
        "ground_truth_urgency": "monitor",
        "image_description": (
            "Two microaneurysms in nasal quadrant. Background retina otherwise clear. "
            "No exudates, no haemorrhages beyond microaneurysms. Early NPDR. "
            "6-month follow-up screening recommended."
        ),
    },
    {
        "patient_id": "P019",
        "patient_age": 66,
        "diabetes_years": 15,
        "ground_truth_label": "moderate_dr",
        "ground_truth_urgency": "refer",
        "image_description": (
            "Moderate NPDR. Multiple haemorrhages and microaneurysms. Hard exudates "
            "in posterior pole. Some cotton wool spots. Macular area shows early "
            "thickening. Ophthalmologist referral within 1 month advised."
        ),
    },
    {
        "patient_id": "P020",
        "patient_age": 70,
        "diabetes_years": 20,
        "ground_truth_label": "proliferative_dr",
        "ground_truth_urgency": "urgent",
        "image_description": (
            "Clear neovascularisation elsewhere on retina. High-risk PDR features. "
            "Pre-retinal haemorrhage. Fibrovascular membranes forming. "
            "Immediate ophthalmology referral for panretinal photocoagulation."
        ),
    },
]


def get_case_by_id(patient_id: str) -> dict:
    for case in PATIENT_CASES:
        if case["patient_id"] == patient_id:
            return case
    raise ValueError(f"Patient {patient_id} not found")


def get_cases_by_label(label: str) -> list:
    return [c for c in PATIENT_CASES if c["ground_truth_label"] == label]
