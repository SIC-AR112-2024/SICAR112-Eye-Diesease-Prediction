def init_prompt_CoT(URL):
    return [
    {'role': 'system',
        'content': """You are a medical student. You will be given several retinal fundus images as a test.
Firstly, describe key features depicted in the image, of no less than 100 words, such as the macula, optic nerve, optic cup and disc and retinal blood discs.
If the eye is healthy, say \"HEALTHY\". If not, tell me whether the patient has \"CATARACT\", \"DIABETIC RETINOPATHY\", or \"GLAUCOMA\". Your final diagnosis must be strictly 1 or 2 words, on a new line."""
    },
    {"role": "user",
        "content": [
            {"type": "text", "text": "Produce a diagnosis for the following:"},
            {"type": "image_url", "image_url": {
                "url": "https://github.com/SIC-AR112-2024/SICAR112-Eye-Diesease-Prediction/blob/main/dataset/diabetic_retinopathy/342.jpg?raw=true",
            },
            },
        ],
    },
    {"role": "assistant",
    "content": """The fundus image shows numerous hard exudates scattered throughout the retina, particularly in the macula. These are yellowish-white deposits that are a hallmark of diabetic retinopathy. Additionally, there are microaneurysms, which are small, dilated blood vessels that leak fluid. These features, along with the presence of retinal hemorrhages, are consistent with diabetic retinopathy. The optic nerve appears healthy, ruling out glaucoma. There is no evidence of a cataract.
DIABETIC RETINOPATHY."""
    },
    {"role": "user",
        "content": [
            {"type": "text", "text": "Produce a diagnosis for the following:"},
            {"type": "image_url", "image_url": {
                "url": "https://github.com/SIC-AR112-2024/SICAR112-Eye-Diesease-Prediction/blob/main/dataset/glaucoma/883.jpg?raw=true",
            },
            },
        ],
    },
    {"role": "assistant",
    "content": """The image shows a view of the fundus of the eye. The optic disc, the point where the optic nerve leaves the eye, is visible in the center of the image. The disc is slightly pale and the margins appear blurred. The macula, the central region responsible for central vision, appears slightly edematous. The retinal blood vessels are clearly visible and appear normal.
GLAUCOMA"""}]

def query_few_shot(query, key):
    ans = ''
    return ans