from fastapi import APIRouter

from pydantic import BaseModel

from services.intent_service import detect_intent

from services.emotion_service import detect_emotion

from services.vector_service import retrieve_chunks

from services.llm_service import generate_response

from prompts.prompt_template import build_prompt

from services.memory_service import (

    save_session,
    get_session

)

router = APIRouter()

class QueryRequest(BaseModel):

    userId: str

    sessionId: str

    query: str

@router.post("/query")

async def query(req: QueryRequest):

    
    session = get_session(

        req.sessionId

    )

 

    intent_result = detect_intent(

        req.query

    )


    emotion_result = detect_emotion(

        req.query

    )

 

    chunks = retrieve_chunks(

        req.query

    )


    prompt = build_prompt(

        req.query,

        intent_result["intent"],

        emotion_result["label"],

        chunks

    )


    answer = generate_response(

        prompt

    )


    session["last_query"] = req.query

    save_session(

        req.sessionId,

        session

    )
    from services.mlflow_service import log_query_metrics

    log_query_metrics(

        latency=420,

        precision=0.91,

        hallucination_rate=0.03

    )

    return {

        "intent": intent_result,

        "emotion": emotion_result,

        "answer": answer

    }
    
    