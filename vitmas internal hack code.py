import streamlit as st
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')





qa_data = [

    # ---------- MEDICAL (5) ----------
    {"category": "medical", "subtype": "location",
     "question": "Where is the health center?",
     "answer": "The Health Center is located in G-Block."},

    {"category": "medical", "subtype": "timing",
     "question": "What are the health center timings?",
     "answer": "The Health Center operates 24/7 including weekends."},

    {"category": "medical", "subtype": "contact",
     "question": "What is the emergency medical contact number?",
     "answer": "Emergency medical contact: 0416-220-2200."},

    {"category": "medical", "subtype": "facility",
     "question": "Does VIT have ambulance service?",
     "answer": "Yes, ambulance service is available on campus for emergencies."},

    {"category": "medical", "subtype": "procedure",
     "question": "Is medical treatment free at VIT?",
     "answer": "Basic consultation is free; medicines and external treatment may be charged."},

     {"category": "medical", "subtype": "location",
     "question": "Where can I buy medicines in VIT?",
     "answer": "All essential medicines are sold in the health center."},

    {"category": "medical", "subtype": "facility",
     "question": "What specialized medical services are available at the Health Center?",
     "answer": "Specialized services include Physiotherapy, Lab tests, X-Ray, and evening consultations for Cardiology, Dermatology, and ENT."},

    {"category": "medical", "subtype": "location",
     "question": "Are there first aid centers in the hostel blocks?",
     "answer": "Yes, first aid centers are located in Men’s Hostel G-Block and Ladies Hostel E-Block."},

    {"category": "medical", "subtype": "facility",
     "question": "What is the capacity of the VIT Health Center?",
     "answer": "The health center has 20 in-patient beds, including separate male/female wards and an isolation ward for infectious diseases."},

    {"category": "medical", "subtype": "timing",
     "question": "What are the Physiotherapy timings at the Health Center?",
     "answer": "Physiotherapy services are available from 11:00 AM to 7:00 PM, Monday to Saturday."},


    # ---------- HOSTEL (10) ----------
    {"category": "hostel", "subtype": "facility",
     "question": "Are hostels available for all students?",
     "answer": "Yes, on-campus hostels are available for both boys and girls."},

    {"category": "hostel", "subtype": "rule",
     "question": "What is the hostel curfew time?",
     "answer": "Curfew timings vary and are announced by the hostel administration."},

    {"category": "hostel", "subtype": "facility",
     "question": "Is there a gym in hostels?",
     "answer": "Yes, gym facilities are available in both men’s and women’s hostels."},

    {"category": "hostel", "subtype": "facility",
     "question": "Is laundry service available?",
     "answer": "Laundry services are available in all hostel blocks."},

    {"category": "hostel", "subtype": "procedure",
     "question": "Can I change my hostel room?",
     "answer": "Room change requests can be submitted during designated periods."},

    {"category": "hostel", "subtype": "rule",
     "question": "Are visitors allowed in hostels?",
     "answer": "Visitors are allowed only in designated areas and timings."},

    {"category": "hostel", "subtype": "facility",
     "question": "Is WiFi available in hostels?",
     "answer": "Yes, WiFi connectivity is available in all hostels."},

    {"category": "hostel", "subtype": "facility",
     "question": "Are AC rooms available?",
     "answer": "Yes, AC and non-AC rooms are available based on allocation."},

    {"category": "hostel", "subtype": "procedure",
     "question": "How do I raise a hostel complaint?",
     "answer": "Hostel complaints can be raised through the VIT grievance portal."},

    {"category": "hostel", "subtype": "facility",
     "question": "Is drinking water available?",
     "answer": "RO drinking water is available on every hostel floor."},

    {"category": "hostel", "subtype": "procedure",
     "question": "How do I book a hostel room at VIT?",
     "answer": "Room booking is done online via the Freshers Portal or VTOP on a first-come, first-served basis after paying the full tuition fee."},

    {"category": "hostel", "subtype": "facility",
     "question": "What items are provided in a hostel room?",
     "answer": "Each student is provided with a cot, chair, study table, and a cupboard."},

    {"category": "hostel", "subtype": "facility",
     "question": "Does VIT have solar water heaters in hostels?",
     "answer": "Yes, all hostel blocks are equipped with solar water heaters and heat pumps to provide hot water."},

    {"category": "hostel", "subtype": "procedure",
     "question": "How do I apply for leave from the hostel?",
     "answer": "Hostel leave can be applied for through the VTOP portal or the VITian mobile app. The students may only leave the campus upon having their leave request verified by their proctor."},

    # ---------- WIFI & TECH (5) ----------
    {"category": "tech", "subtype": "procedure",
     "question": "How do I connect to VIT WiFi?",
     "answer": "Use your registration number as username and VTOP password."},

    {"category": "tech", "subtype": "troubleshooting",
     "question": "WiFi is not working, what should I do?",
     "answer": "Restart your device and reconnect. If it persists, contact IT support."},

    {"category": "tech", "subtype": "contact",
     "question": "How do I contact IT support?",
     "answer": "IT support can be contacted through the VIT helpdesk portal."},

    {"category": "tech", "subtype": "rule",
     "question": "Is WiFi usage monitored?",
     "answer": "Yes, WiFi usage is monitored and subject to VIT IT policies."},

    {"category": "tech", "subtype": "procedure",
     "question": "Can I use my own router?",
     "answer": "Personal routers are generally not allowed in hostels."},

    {"category": "tech", "subtype": "info",
     "question": "What is VTOP?",
     "answer": "VTOP (VIT on TOP) is the official digital platform for students to manage academics, attendance, fees, and placement activities."},

    {"category": "tech", "subtype": "contact",
     "question": "Who manages the IT infrastructure at VIT?",
     "answer": "The Centre for Technical Support (CTS) maintains the campus-wide network, WiFi, and IT policies."},

    {"category": "tech", "subtype": "info",
     "question": "Is there a mobile app for VIT students?",
     "answer": "Yes, the 'VITian' app allows students to check attendance, timetables, and apply for hostel leave."},

    # ---------- ACADEMICS & ATTENDANCE (10) ----------
    {"category": "academics", "subtype": "rule",
     "question": "What is the minimum attendance required?",
     "answer": "A minimum of 75% attendance is mandatory."},

    {"category": "academics", "subtype": "consequence",
     "question": "What happens if attendance is below 75 percent?",
     "answer": "Students may be debarred from appearing in FAT."},

    {"category": "academics", "subtype": "procedure",
     "question": "Can I apply for attendance condonation?",
     "answer": "Condonation may be approved under exceptional circumstances."},

    {"category": "academics", "subtype": "info",
     "question": "What is FFCS?",
     "answer": "FFCS allows students to choose courses, faculty, and time slots."},

    {"category": "academics", "subtype": "info",
     "question": "What is CAT?",
     "answer": "CAT stands for Continuous Assessment Test."},

    {"category": "academics", "subtype": "info",
     "question": "What is FAT?",
     "answer": "FAT is the Final Assessment Test conducted at semester end."},

    {"category": "academics", "subtype": "procedure",
     "question": "How do I register for courses?",
     "answer": "Course registration is done through VTOP during the registration window."},

    {"category": "academics", "subtype": "info",
     "question": "What is a backlog?",
     "answer": "A backlog is a course that has to be reattempted after failing."},

    {"category": "academics", "subtype": "procedure",
     "question": "Can I drop a course?",
     "answer": "Courses can be dropped within the add/drop deadline."},

    {"category": "academics", "subtype": "info",
     "question": "How many credits per semester?",
     "answer": "A typical semester has 24–27 credits."},

    {"category": "academics", "subtype": "rule",
     "question": "What is the slot-based timetable in FFCS?",
     "answer": "A slot-based timetable allows students to choose specific time slots for theory and lab classes to avoid overlaps."},

    {"category": "academics", "subtype": "procedure",
     "question": "What is the Add/Drop option in FFCS?",
     "answer": "It is a window during the semester where students can add a new course or drop an existing one to adjust their workload."},

    {"category": "academics", "subtype": "info",
     "question": "What is PBL at VIT?",
     "answer": "PBL stands for Project Based Learning, where students apply theoretical principles to real-world engineering problems through projects."},


    # ---------- MESS (5) ----------
    {"category": "mess", "subtype": "timing",
     "question": "What are mess timings?",
     "answer": "Breakfast 7–9 AM, Lunch 12:30–2 PM, Snacks 4:30–5:30 PM, Dinner 7:30–9 PM."},

    {"category": "mess", "subtype": "facility",
     "question": "Is vegetarian food available?",
     "answer": "Yes, vegetarian food is available in all messes."},

    {"category": "mess", "subtype": "procedure",
     "question": "Can I change my mess?",
     "answer": "Mess change is allowed during the mess change window."},

    {"category": "mess", "subtype": "rule",
     "question": "Are outsiders allowed in mess?",
     "answer": "Only registered students are allowed inside mess halls."},

    {"category": "mess", "subtype": "facility",
     "question": "Is night mess available?",
     "answer": "Night mess is available in selected hostel areas."},

    {"category": "mess", "subtype": "facility",
     "question": "Where is the GDN Canteen located?",
     "answer": "The GDN Canteen is located in the Gandhi Block and offers budget-friendly snacks and South Indian lunch."},

    {"category": "mess", "subtype": "facility",
     "question": "Are there private food courts on campus?",
     "answer": "Yes, there are several private dining options like FC Food Court and Limra Student Restaurant located within or near campus areas."},


    # ---------- CAMPUS LOCATIONS (10) ----------
    {"category": "campus", "subtype": "location",
     "question": "Where is the library?",
     "answer": "The Central Library is near the main gate."},

    {"category": "campus", "subtype": "location",
     "question": "Where is the main gate?",
     "answer": "The Main Gate is on Katpadi Road."},

    {"category": "campus", "subtype": "location",
     "question": "Where is Anna Auditorium?",
     "answer": "Anna Auditorium is located near the academic blocks."},

    {"category": "campus", "subtype": "location",
     "question": "Where is the food street?",
     "answer": "The food street is near the men’s hostel area."},

    {"category": "campus", "subtype": "location",
     "question": "Where is the post office?",
     "answer": "The post office is near the administrative buildings."},

    {"category": "campus", "subtype": "location",
     "question": "Where are the banks and ATMs?",
     "answer": "Banks and ATMs are available across campus near hostels and academic blocks."},

    {"category": "campus", "subtype": "location",
     "question": "Where is the sports complex?",
     "answer": "The sports complex is located near the outdoor stadium."},

    {"category": "campus", "subtype": "location",
     "question": "Where is the swimming pool?",
     "answer": "The swimming pool is located inside the sports complex."},

    {"category": "campus", "subtype": "location",
     "question": "Where is the administrative block?",
     "answer": "The administrative block is near the main academic buildings."},

    {"category": "campus", "subtype": "location",
     "question": "Where is the girls’ hostel area?",
     "answer": "Girls’ hostels are located on the eastern side of the campus."},
    {"category": "placements", "subtype": "info",
     "question": "What is the CDC?",
     "answer": "CDC stands for Career Development Centre, which handles placements and internships."},

    {"category": "campus", "subtype": "timing",
     "question": "What are the Central Library working hours?",
     "answer": "The library is open daily: Mon-Fri (7:30 AM – 12:30 AM), and Sat-Sun (9:00 AM – 8:00 PM)."},

    {"category": "campus", "subtype": "location",
     "question": "Where is the Career Development Centre (CDC)?",
     "answer": "The CDC is located near the main administrative area and handles all placement-related activities."},

    {"category": "campus", "subtype": "facility",
     "question": "Is there a post office on campus?",
     "answer": "Yes, a post office is available near the administrative buildings for student and staff use."},

   #---------------------CLUBS AND EVENTS--------------------
    {"category": "events", "subtype": "info",
     "question": "What is Riviera?",
     "answer": "Riviera is VIT’s annual international sports and cultural carnival."},

    {"category": "events", "subtype": "info",
     "question": "What is graVITas?",
     "answer": "graVITas is VIT’s annual technological and design festival."},

    {"category": "events", "subtype": "procedure",
     "question": "How can I join a club?",
     "answer": "Students can join clubs during the club recruitment season at the start of the semester."},

    {"category": "events", "subtype": "location",
     "question": "Where are club activities usually held?",
     "answer": "Most activities happen in the academic blocks or the outdoor plazas."},

    {"category": "events", "subtype": "rule",
     "question": "Can I start my own club?",
     "answer": "Yes, new clubs can be proposed to the Office of Student Welfare (OSW)."},

    #===================FINANCE============================
    {"category": "finance", "subtype": "procedure",
     "question": "How can I pay my tuition fees?",
     "answer": "Tuition fees are paid online through the VTOP student portal."},

    {"category": "finance", "subtype": "info",
     "question": "Are there scholarships available?",
     "answer": "Yes, VIT offers scholarships like GVSET and merit-based fee waivers."},

    {"category": "finance", "subtype": "procedure",
     "question": "How do I get a fee structure for an education loan?",
     "answer": "The official fee structure can be downloaded from VTOP or obtained from the accounts office."},

    {"category": "finance", "subtype": "consequence",
     "question": "What happens if I pay fees late?",
     "answer": "Late payments usually incur a fine as per the university's financial policy."},

    {"category": "finance", "subtype": "location",
     "question": "Where is the accounts office?",
     "answer": "The accounts office is located within the Administrative Block."},

    #=============================TRANSPORT AND OUTING=============================
    {"category": "transport", "subtype": "facility",
     "question": "Is there a shuttle service inside campus?",
     "answer": "Yes, shuttle buses and electric carts operate across major campus points."},

    {"category": "transport", "subtype": "rule",
     "question": "Are day scholars allowed to bring vehicles?",
     "answer": "Yes, but they must register their vehicle and obtain a VIT parking sticker."},

    {"category": "transport", "subtype": "procedure",
     "question": "How do I apply for an outing?",
     "answer": "Outing permissions are applied for through the VIT Hostel App or VTOP."},

    {"category": "transport", "subtype": "rule",
     "question": "Can hostelers keep motorized vehicles?",
     "answer": "No, hostellers are generally not permitted to keep motorized vehicles on campus."},

    {"category": "transport", "subtype": "location",
     "question": "Where is the nearest railway station?",
     "answer": "The nearest station is Katpadi Junction, located about 3 km from the campus."}

]

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()

INTENTS = {
    "medical_emergency": [
        "headache", "pain", "fever", "vomit", "dizzy", "injury",
        "bleeding", "doctor", "medical", "emergency", "ambulance"
    ],
    "hostel_issue": ["hostel", "room", "gym", "laundry", "warden"],
    "wifi_issue": ["wifi", "internet", "network"],
    "academic_issue": ["attendance", "exam", "ffcs", "fat", "cat"],
    "mess_issue": ["food", "mess", "breakfast", "lunch", "dinner"],
    "campus_navigation": ["where", "located", "location", "near"],
}

INTENT_TO_CATEGORY = {
    "medical_emergency": "medical",
    "hostel_issue": "hostel",
    "wifi_issue": "tech",
    "academic_issue": "academics",
    "mess_issue": "mess",
    "campus_navigation": "campus",
}

FORCE_ALL = {
    "medical_emergency": ["location", "timing", "contact", "facility", "procedure"]
}

def detect_intent(text):
    scores = {intent: 0 for intent in INTENTS}
    for intent, keywords in INTENTS.items():
        for kw in keywords:
            if kw in text:
                scores[intent] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"

def retrieve_answers(user_input, threshold=0.15):
    cleaned = clean_text(user_input)
    intent = detect_intent(cleaned)
    category = INTENT_TO_CATEGORY.get(intent)

    if intent in FORCE_ALL:
        answers = []
        for subtype in FORCE_ALL[intent]:
            answers.extend([
                qa["answer"]
                for qa in qa_data
                if qa["category"] == category and qa["subtype"] == subtype
            ])
        return intent, answers

    relevant = [qa for qa in qa_data if qa["category"] == category] if category else qa_data
    if not relevant:
        return intent, []

    questions = [qa["question"] for qa in relevant]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(questions + [cleaned])

    scores = cosine_similarity(vectors[-1], vectors[:-1])[0]
    ranked = sorted(zip(relevant, scores), key=lambda x: x[1], reverse=True)

    answers = [qa["answer"] for qa, s in ranked if s >= threshold][:3]
    return intent, answers

def format_response(intent, answers):
    if not answers:
        return "I couldn’t find a precise answer. Please contact the Health Center if this is urgent."

    header_map = {
        "medical_emergency": "🏥 Medical Assistance",
        "hostel_issue": "🏠 Hostel Information",
        "mess_issue": "🍽 Mess Information",
        "campus_navigation": "📍 Campus Location",
    }

    header = header_map.get(intent, "ℹ Information")
    lines = [header]

    for ans in dict.fromkeys(answers):
        lines.append("• " + ans)

    return "\n".join(lines)



if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

st.sidebar.title("Settings")
st.session_state.dark_mode = st.sidebar.checkbox("Dark Mode", value=st.session_state.dark_mode)
with st.sidebar.expander("FAQ"):
    st.markdown("""
    **Q1:** What is the emergency medical contact number?\n 
    **A1:** Emergency medical contact: 0416-220-2200.

    **Q2:** Is there a gym in hostels?\n  
    **A2:** Yes, gym facilities are available in both men’s and women’s hostels."

    **Q3:** What are mess timings?\n 
    **A3:** Breakfast 7–9 AM, Lunch 12:30–2 PM, Snacks 4:30–5:30 PM, Dinner 7:30–9 PM.
    """)
with st.sidebar.expander("Help"):
    st.markdown("""
    **Q1:** How do I use CLANKER?  
    **A1:** Type your query in the input box and hit enter.

    **Q2:** Can I exit the app?  
    **A2:** Type 'bye' or 'exit', and the app will stop responding.

    **Q3:** How do I toggle dark/light mode?  
    **A3:** Use the checkbox above in the sidebar.
    """)




dark_css = """
<style>
.stApp {
    background-color: #0E1117;
    color: #FFFFFF;
}
section[data-testid="stSidebar"] {
    background-color: #111827;
    color: #FFFFFF;
}
h1, h2, h3, h4, h5, h6, p {
    color: #FFFFFF;
}
</style>
"""

light_css = """
<style>
.stApp {
    background-color: #FFFFFF;
    color: #000000;
}
section[data-testid="stSidebar"] {
    background-color: #F0F2F6;
    color: #000000;
}
h1, h2, h3, h4, h5, h6, p {
    color: #000000;
}
</style>
"""

st.markdown(dark_css if st.session_state.dark_mode else light_css, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>CLANKER</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>The Freshman Survival Database</h3>", unsafe_allow_html=True)

# Initialize state
if "started" not in st.session_state:
    st.session_state.started = False

left, center, right = st.columns([3, 1, 3])

with center:
    st.markdown("""<style>
    div.stButton > button {
        width: auto;
        height: auto;
        font-size: 18px;
        border-radius: 10px;
        border: none;
        transition: all 0.2s;
    }
    </style>""", unsafe_allow_html=True)

    if st.button("Click here to ask your query"):
        st.session_state.started = True

# List of exit commands
l = ['bye','exit','thanks','thank you']

if "q_count" not in st.session_state:
    st.session_state.q_count = 0
if "history" not in st.session_state:
    st.session_state.history = []

if st.session_state.started:
    user_input = st.text_input("enter your query", key=st.session_state.q_count)

    if user_input:
        if user_input.lower() in l:
            st.write("Thank you")
            st.stop()
        

        else:
            # 🔹 ONLY INTEGRATION LINE 🔹
            intent, answers = retrieve_answers(user_input)
            st.write(format_response(intent, answers))

            st.session_state.history.append(user_input)
            st.session_state.q_count += 1

bg_color = "#111827" if st.session_state.dark_mode else "#f0f2f6"
text_color = "#FFFFFF" if st.session_state.dark_mode else "#000000"

st.markdown(f"""
<style>
.footer {{
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    width: 100%;
    background-color: {bg_color};
    color: {text_color};
    text-align: center;
    padding: 10px;
    font-size: 18px;
    font-weight: bold;
    z-index: 9999;
}}
</style>
<div class="footer">
    This is an AI application and hence is liable to certain inaccuracies.
</div>
""", unsafe_allow_html=True)



            

        
        

