import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import google.generativeai as genai
import os
import numpy as np
import whisper
import tempfile
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase

# ✅ FFmpeg Path (Ensure Correct Path for Your OS)
FFMPEG_PATH = "C:\\ffmpeg\\bin\\ffmpeg.exe"  # Windows
# FFMPEG_PATH = "/usr/bin/ffmpeg"  # Linux/macOS
os.environ["PATH"] += os.pathsep + os.path.dirname(FFMPEG_PATH)

# ✅ Streamlit Page Configuration
st.set_page_config(page_title="Skin Disease Chatbot", layout="centered")

# ✅ Configure Google Gemini API (read from env)
# Try .env/env first, then Streamlit secrets (safely)
try:
	from dotenv import load_dotenv
	load_dotenv()
except Exception:
	pass

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or ""
if not GEMINI_API_KEY:
	try:
		GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
	except Exception:
		GEMINI_API_KEY = "AIzaSyALBGc5kg_6wLwb3dpRlF5z5caRB2fintc"

# Provide a thin wrapper so code paths can call a uniform function
class _GeminiClient:
	def __init__(self, api_key: str):
		self._model = None
		self._version = None
		self._new_candidates = [
			"gemini-1.5-flash",
			"gemini-1.5-flash-8b",
			"gemini-1.5-pro",
			"gemini-pro",
		]
		self._legacy_candidates = [
			"models/chat-bison-001",
			"models/text-bison-001",
		]
		try:
			genai.configure(api_key=api_key)
			# Prefer new API (>= 0.3.x): GenerativeModel class
			if hasattr(genai, "GenerativeModel"):
				try:
					self._model = genai.GenerativeModel(self._new_candidates[0])
					self._version = "new"
				except Exception:
					# Defer actual generation to fallback logic
					self._model = None
					self._version = "new"
			else:
				# Legacy 0.1.0rc1: try known PaLM model ids
				self._model = self._legacy_candidates[0]
				self._version = "legacy"
		except Exception as e:
			st.error(f"Failed to initialize Gemini model: {e}")
			self._model = None

	@property
	def is_ready(self) -> bool:
		return self._version is not None

	def _offline_response(self, prompt: str) -> str:
		# Minimal safe fallback if online generation fails
		generic = (
			"I'm currently offline. General advice: keep the area clean and dry, avoid scratching, "
			"and consider over-the-counter hydrocortisone for itch unless you have open wounds. "
			"If symptoms worsen, spread rapidly, or involve fever, seek medical care promptly."
		)
		try:
			if "User has" in prompt and "Provide a medical treatment suggestion" in prompt:
				# Image flow prompt
				start = prompt.find("User has ") + len("User has ")
				end = prompt.find(" based on an image")
				disease = prompt[start:end].strip() if end > start else "a skin condition"
				return f"Based on the provided image, a likely diagnosis is {disease}. {generic}"
			else:
				# Symptom text flow prompt
				return f"Based on the described symptoms, here is general guidance: {generic}"
		except Exception:
			return generic

	def generate(self, prompt: str) -> str:
		if not self.is_ready:
			return self._offline_response(prompt)
		try:
			if self._version == "new":
				# Try current/new candidates sequentially
				candidates = self._new_candidates if self._new_candidates else ["gemini-1.5-flash"]
				last_err = None
				for model_id in candidates:
					try:
						model = genai.GenerativeModel(model_id)
						resp = model.generate_content(prompt)
						self._model = model
						return getattr(resp, "text", str(resp))
					except Exception as e:
						last_err = e
						msg = str(e)
						# On permission/404/not found, continue to next candidate
						if any(s in msg.lower() for s in ["404", "not found", "permission", "model"]):
							continue
						# Other errors: break and fallback
						break
				# If all candidates failed, provide offline response with last error note
				return self._offline_response(prompt)
			else:
				# Legacy API: genai.generate_text(model=..., prompt=...)
				try:
					resp = genai.generate_text(model=self._model, prompt=prompt)
					return getattr(resp, "result", str(resp))
				except Exception as e:
					# If 404 or not found, try next candidate model
					msg = str(e)
					if "404" in msg or "not found" in msg.lower():
						for candidate in self._legacy_candidates:
							if candidate == self._model:
								continue
							try:
								resp = genai.generate_text(model=candidate, prompt=prompt)
								self._model = candidate
								return getattr(resp, "result", str(resp))
							except Exception:
								continue
					# Otherwise offline fallback
					return self._offline_response(prompt)
		except Exception:
			return self._offline_response(prompt)

if not GEMINI_API_KEY:
	st.warning("GEMINI_API_KEY not set. Set it in your environment or .env to enable Gemini responses.")
	gemini = None
else:
	gemini = _GeminiClient(GEMINI_API_KEY)

# ✅ Disease Classes
classes = ["Actinic keratoses", "Basal cell carcinoma", "Benign keratosis-like lesions",
           "Chickenpox", "Cowpox", "Dermatofibroma", "Healthy", "HFMD", "Measles",
           "Melanocytic nevi", "Melanoma", "Monkeypox", "Squamous cell carcinoma", "Vascular lesions"]

# ✅ Load Models Once
@st.cache_resource
def load_models():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model_paths = {
		"resnet50": "trained_models/resnet50.pth",
		"efficientnet_b3": "trained_models/efficientnet_b3.pth",
		"densenet121": "trained_models/densenet121.pth"
	}
	models_dict = {}

	for name, path in model_paths.items():
		if os.path.exists(path):
			model = models.__dict__[name](weights=None)
			if name == "resnet50":
				model.fc = torch.nn.Linear(model.fc.in_features, 14)
			elif name == "efficientnet_b3":
				model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 14)
			elif name == "densenet121":
				model.classifier = torch.nn.Linear(model.classifier.in_features, 14)

			model.load_state_dict(torch.load(path, map_location=device))
			model.eval()
			models_dict[name] = model.to(device)
		else:
			st.warning(f"⚠ Model file {path} not found. Please train and place it in 'trained_models/'.")
	
	return models_dict

ensemble_models = load_models()

# ✅ Image Preprocessing
transform = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ✅ Predict Skin Disease from Image
def predict_image(image):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	img_tensor = transform(image).unsqueeze(0).to(device)

	predictions = []
	with torch.no_grad():
		for model in ensemble_models.values():
			output = model(img_tensor)
			probs = torch.softmax(output, dim=1).cpu().numpy()[0]
			predictions.append(probs)

	avg_probs = np.mean(predictions, axis=0) if predictions else np.zeros(14)
	pred_idx = int(np.argmax(avg_probs)) if predictions else 0
	disease = classes[pred_idx]

	if not gemini or not gemini.is_ready:
		return disease, "Gemini is not configured. Set GEMINI_API_KEY to enable suggestions."

	prompt = f"User has {disease} based on an image. Provide a medical treatment suggestion in a friendly chatbot style."
	response_text = gemini.generate(prompt)

	return disease, response_text

# ✅ Process Text Input with Gemini
def process_text(text):
	if not gemini or not gemini.is_ready:
		return "Gemini is not configured. Set GEMINI_API_KEY to enable suggestions."
	prompt = f"Given these symptoms: '{text}', predict the most likely skin disease from: {classes}. Provide a treatment suggestion."
	return gemini.generate(prompt)

# ✅ Load Whisper Model Once
whisper_model = whisper.load_model("base")

# ✅ Process Audio and Get Transcription
def process_audio(audio_data):
	"""Transcribe speech to text and predict disease from symptoms."""

	with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
		temp_audio.write(audio_data)
		temp_audio_path = temp_audio.name

	result = whisper_model.transcribe(temp_audio_path)
	transcribed_text = result["text"]

	os.remove(temp_audio_path)

	# ✅ Pass transcribed text to Gemini for disease prediction
	prediction = process_text(transcribed_text)

	return transcribed_text, prediction

# ✅ Maintain Chat History in Session State
if "messages" not in st.session_state:
	st.session_state.messages = []
if "last_input_type" not in st.session_state:
	st.session_state.last_input_type = None  # Track previous input type

st.title("🩺 Skin Disease Chatbot")
st.write("👋 Upload an image, describe your symptoms, or *record your voice* to get a diagnosis & suggestion!")

# ✅ Clear Previous Messages When Changing Input Type
input_type = st.selectbox("Choose an Input Method:", ["Text", "Image", "Audio"])
if input_type != st.session_state.last_input_type:
	st.session_state.messages.clear()
	st.session_state.last_input_type = input_type  # Update the last input type

# ✅ Display Chat History
for message in st.session_state.messages:
	with st.chat_message(message["role"]):
		st.markdown(message["content"])

# 📜 **Text Input Mode**
if input_type == "Text":
	user_text = st.chat_input("Describe your symptoms (e.g., 'red itchy rash'):")
	if user_text:
		with st.chat_message("user"):
			st.markdown(user_text)

		with st.chat_message("assistant"):
			response = process_text(user_text)
			st.markdown(response)

		st.session_state.messages.append({"role": "user", "content": user_text})
		st.session_state.messages.append({"role": "assistant", "content": response})

# 🖼 **Image Input Mode**
elif input_type == "Image":
	uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "png"])
	if uploaded_file:
		image = Image.open(uploaded_file).convert("RGB")
		st.image(image, caption="Your Image", width=250)

		with st.chat_message("user"):
			st.markdown("Here's my skin photo!")

		with st.chat_message("assistant"):
			disease, response = predict_image(image)
			st.markdown(f"🩺 *Diagnosis:* {disease}\n\n💡 *Suggestion:* {response}")

		st.session_state.messages.append({"role": "user", "content": "Here's my skin photo!"})
		st.session_state.messages.append({"role": "assistant", "content": f"*Diagnosis:* {disease}\n\n*Suggestion:* {response}"})

# 🎤 **Audio Recording Mode**
elif input_type == "Audio":
	uploaded_audio = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
	if uploaded_audio:
		audio_bytes = uploaded_audio.read()

		with st.spinner("Transcribing & Predicting..."):
			transcription, prediction = process_audio(audio_bytes)

		st.markdown(f"📝 **Transcription:** {transcription}")
		st.markdown(f"💡 **Prediction & Suggestion:** {prediction}")

		st.session_state.messages.append({"role": "user", "content": transcription})
		st.session_state.messages.append({"role": "assistant", "content": prediction})
