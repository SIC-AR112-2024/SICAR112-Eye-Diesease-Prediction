def query_0_shot(query, key):
    ans = ''
    return ans

def init_prompt_zero_shot(URL):
    return [
    {'role':'user',
    'content':[
        {'type': 'text', 'text':'''You are a medical student. You will be given several retinal fundus images as a test.
Firstly, describe key features depicted in the image, of no less than 100 words, such as the macula, optic nerve, optic cup and disc and retinal blood discs.
If the eye is healthy, say "HEALTHY". If not, tell me whether the patient has "CATARACT", "DIABETIC RETINOPATHY", or "GLAUCOMA". Your final diagnosis must be strictly 1 or 2 words, on a new line.'''},
        {'type': 'image_url', 'image_url':{'url': URL}}
    ]}]