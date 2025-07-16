from llama_cpp import Llama, LlamaGrammar
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json


class Query(BaseModel):
    text: str


class chatbot_extracter:
    def __init__(self, model_path, context_size, n_threads, message_layout, grammar):
        self.m_layout = message_layout
        self.grammar = LlamaGrammar.from_string(grammar)
        self.llm = Llama(
            model_path=model_path,
            n_ctx=context_size,
            n_threads=n_threads,
            verbose=False,
        )

    async def extract_heating_data(self, query: Query):
        try:
            prompt = self.m_layout(query.text)
            response = self.llm(
                prompt, temperature=0, max_tokens=128, grammar=self.grammar, stop=["}"]
            )
            json_str = response["choices"][0]["text"] + "}"
            return JSONResponse(json.loads(json_str))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    # Configuration
    MODEL_PATH = "models/Qwen3-0.6B-Q5_K_M.gguf"  # Path to your GGUF file
    N_CTX = 2048  # Context window size
    N_THREADS = 8  # CPU threads


    def build_prompt(text):
        return f"""
        Example 1:
        Input: "Fjernvarme med isoleret veksler (indirekte anlæg) - nyere. 
        Bygningen opvarmes med fjernvarme. 
        Anlægget er udført med isoleret varmeveksler og indirekte centralvarmevand i fordelingsnettet. 
        Anlægget er opstillet i Badensgade 41."
        Output: {{
            "Pieces": 1,
            "Manufacturer": "Unknown",
            "SubType": null,
            "HxType": "Eksisterende fjernvarme",
            "NominalEffectEach": null,
            "Year": null
        }}

        Example 2:
        Input: "Fjernvarme med isoleret veksler (indirekte anlæg) - nyere. Ejendommen opvarmes med fjernvarme fra HOFOR.
    Til opvarmning af radiatorerne er der 1 stk. isoleret varmevekslere monteret i fjernvarmeunit,  fabrikat Redan.
    Fjernvarmeunit er placeret i kælderen."
        Output: {{
            "Pieces": 1,
            "Manufacturer": "Danfoss Redan",
            "SubType": null,
            "HxType": "Isoleret varmeveksler",
            "NominalEffectEach": null,
            "Year": null
        }}

        Example 3:
        Input: "Fjv. Installation efter 1980 (isoleret). Ejendommen opvarmes med indirekte fjernvarme. 
    Bygningen opvarmes med fjernvarme med veksler.
    Veksleren er af fabrikat WPH, type SL70TL-1-90CC fra 2008 og vurderes isoleret med 40 mm PUR.
    Jf. tidligere energimærkerapport er der brændeovne i de enkelte boliger.
    I værkstedet i baghuset mod nordvest er der monteret en lille elradiator. Rummet er ikke medtaget som opvarmet i beregningen, da rummet alene vurderes kortvarigt opvarmet."
        Output: {{
            "Pieces": 1,
            "Manufacturer": "WPH Teknik",
            "SubType": SL70TL-1-90CC,
            "HxType": "Isoleret varmeveksler",
            "NominalEffectEach": null,
            "Year": 2008
        }}

        Example 4:
        Input: "Fjernvarme med isoleret veksler (indirekte anlæg) - efter 1980. Bygningen opvarmes med fjernvarme.
    Anlægget er udført med isoleret varmeveksler og indirekte centralvarmevand i fordelingsnettet.
    I teknikrum er opstillet 2 stk. varmevekslere - 1 stk. fabr. Sondex type ST 15-ST (radiatorer) à 200 kW og 1 stk. Sondex type ST 20-ST (ventilationsvarmeflader) à 180 kW.
    Vekslere er præisolerede."
        Output: {{
            "Pieces": 1,
            "Manufacturer": "Sondex Teknik",
            "SubType": "ST 15-ST",
            "HxType": "Isoleret varmeveksler",
            "NominalEffectEach": "200 kW",
            "Year": "After 1980"
        }}

        You are provided a sentence, and you have to extract the following values:
            - pieces: the quantity of heating systems.
            - manufacturer: if it is not specified it is Unknown
            - SubType: must be id to identify the model, e.g. null, SL3323TLX, SKR, APVB,
            - HxType: Heat Exchanger Type, e.g. Isoleret varmeveksler, Fjernvarmeveksler, Varmeveksler
            - NominalEffectEach, e.g. 1100 kW, 150 kW, 400 kW
            - Year: e.g. After 1980, 2017, 2000-2009
        Return a JSON with the values.
        Input: "{text}"
        Output:
        """


    GRAMMAR = r"""
    root ::= "{" p "," m "," s "," h "," n "," y "}"
    p ::= "\"Pieces\": " (number | "null")
    number ::= [0-9]+ ("." [0-9]+)?
    m ::= "\"Manufacturer\": " ("\"Unknown\"" | "\"" [a-zA-Z0-9æøåÆØÅ ]+ "\"")
    s ::= "\"SubType\": " ("null" | "\"" [a-zA-Z0-9æøåÆØÅ ]+ "\"")
    h ::= "\"HxType\": " ("null" | "\"" [a-zA-Z0-9æøåÆØÅ ]+ "\"")
    n ::= "\"NominalEffectEach\": " ("null" | "\"" + number + " " ("kW" | "MW" | "W") + "\"")
    y ::= "\"Year\": " ("null" | "\"" [a-zA-Z0-9æøåÆØÅ ]+ "\"" | [0-9]{4})
    """

    app = FastAPI()

    ce = chatbot_extracter(MODEL_PATH, N_CTX, N_THREADS, build_prompt, GRAMMAR)


    @app.post("/extract")
    async def extract_heating_data(query: Query):
        try:
            return await ce.extract_heating_data(query)
        except Exception as e:
            print(f"Error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    uvicorn.run(app, host="0.0.0.0", port=8000)
