import os
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from llama_cpp import Llama

# ============================================================
#                    CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "Chroma_db")
MODEL_PATH = os.path.join(BASE_DIR, "Models", "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf")

# Parametres du modele
N_CTX = 2048        # Taille du contexte
N_THREADS = 4       # Nombre de threads CPU
N_BATCH = 256       # Taille du batch
MAX_TOKENS = 400    # Longueur max de la reponse
TEMPERATURE = 0.1   # Creativite (0.1 = tres factuel)
TOP_K = 5           # Nombre de documents a recuperer

# ============================================================
#                    CHARGEMENT DES COMPOSANTS
# ============================================================

def load_embeddings():
    """Charge le modele d'embeddings multilingual-e5"""
    print("\n[1/3] Chargement des embeddings...")
    start = time.time()
    emb = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print(f"  -> Embeddings charges en {time.time() - start:.1f}s")
    return emb


def load_database(embeddings):
    """Connecte la base ChromaDB"""
    print("\n[2/3] Connexion a ChromaDB...")
    
    if not os.path.exists(DB_DIR):
        print(f"  ERREUR : Le dossier {DB_DIR} n'existe pas !")
        print(f"  Lancez d'abord : python ingestion_chroma.py")
        exit()
    
    database = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )
    print(f"  -> Base connectee")
    return database


def load_llm():
    """Charge le modele Llama-3"""
    print("\n[3/3] Chargement de Llama-3 (CPU)...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"  ERREUR : Le modele n'existe pas : {MODEL_PATH}")
        exit()
    
    start = time.time()
    model = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_batch=N_BATCH,
        verbose=False
    )
    print(f"  -> Llama-3 charge en {time.time() - start:.1f}s")
    return model


# ============================================================
#                    DETECTION DE LANGUE
# ============================================================

def detect_language(text):
    """Detecte si le texte est en arabe ou en francais"""
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    total = len(text.strip())
    if total == 0:
        return "fr"
    return "ar" if arabic_chars > total * 0.3 else "fr"


# ============================================================
#                    RECHERCHE DE DOCUMENTS
# ============================================================

def search_documents(database, query, k=TOP_K):
    """Recherche les documents les plus pertinents dans ChromaDB"""
    # Le prefixe "query: " est obligatoire pour le modele E5
    results = database.similarity_search("query: " + query, k=k)
    return results


def format_context(documents):
    """Formate les documents recuperes en un seul texte de contexte"""
    context_parts = []
    for i, doc in enumerate(documents):
        text = doc.page_content.replace("passage: ", "")
        source = os.path.basename(doc.metadata.get("source", "inconnu"))
        page = doc.metadata.get("page", "?")
        context_parts.append(f"[Document {i+1} - {source} page {page}]\n{text}")
    return "\n\n".join(context_parts)


# ============================================================
#                    CONSTRUCTION DU PROMPT
# ============================================================

def build_prompt(query, documents):
    """Construit le prompt au format officiel Llama-3"""
    context = format_context(documents)
    lang = detect_language(query)

    if lang == "ar":
        system_msg = """أنت مساعد خبير في الخزينة العامة للمملكة المغربية.
التعليمات:
- أجب باللغة العربية فقط
- استخدم فقط المعلومات الموجودة في السياق المقدم
- كن مختصراً ودقيقاً
- اذكر مصدر المعلومة عند الإمكان
- إذا لم تجد الإجابة في السياق، قل: لم أجد هذه المعلومة في الوثائق المتاحة
- لا تكرر الإجابة"""

        user_msg = f"""السياق:
{context}

السؤال: {query}"""

    else:
        system_msg = """Vous etes un assistant expert de la TGR (Tresorerie Generale du Royaume du Maroc).
Instructions:
- Repondez directement a la question, sans vous presenter
- Utilisez uniquement les informations du contexte fourni
- Soyez precis et concis
- Citez le document source quand c'est possible
- Si vous ne trouvez pas la reponse, dites: Information non trouvee dans les documents
- Ne repetez pas la reponse
- Ne dites jamais Cordialement ou Merci"""

        user_msg = f"""Contexte:
{context}

Question: {query}"""

    # Format officiel Llama-3 Instruct
    prompt = f"""<|start_header_id|>system<|end_header_id|>

{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return prompt


# ============================================================
#                    GENERATION DE REPONSE
# ============================================================

# Stop tokens pour eviter les repetitions
STOP_TOKENS = [
    "<|eot_id|>",
    "<|end_header_id|>",
    "Question:",
    "Question :",
    "السؤال:",
    "Cordialement",
    "Note :",
    "Merci",
    "[Votre nom]",
    "Vous :",
    "\n\n\n",
    "Reponse :",
    "Réponse :",
    "الجواب:",
    "___",
    "Assistant expert",
]


def generate_response(model, prompt):
    """Genere une reponse avec Llama-3"""
    output = model(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=0.9,
        repeat_penalty=1.2,
        stop=STOP_TOKENS
    )
    response = output["choices"][0]["text"].strip()
    return clean_response(response)


def clean_response(text):
    """Nettoie la reponse generee"""
    if not text:
        return "Aucune reponse generee."
    
    # Supprimer les caracteres residuels
    text = text.rstrip("(").strip()
    
    # Supprimer les lignes de tirets ou vides
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("_____"):
            continue
        if stripped.startswith("Note :"):
            break
        if stripped.startswith("Cordialement"):
            break
        if stripped.startswith("Merci"):
            break
        if stripped.startswith("[Votre"):
            break
        if stripped.startswith("Assistant expert"):
            break
        cleaned_lines.append(line)
    
    text = "\n".join(cleaned_lines).strip()
    
    # Si la reponse est vide apres nettoyage
    if not text:
        return "Information non trouvee dans les documents disponibles."
    
    return text


# ============================================================
#                    AFFICHAGE DES SOURCES
# ============================================================

def show_sources(documents):
    """Affiche les sources des documents retrouves sans doublons"""
    seen = set()
    sources = []
    for doc in documents:
        source = os.path.basename(doc.metadata.get("source", "inconnu"))
        page = doc.metadata.get("page", "?")
        key = f"{source}_p{page}"
        if key not in seen:
            seen.add(key)
            sources.append(f"   -> {source} (page {page})")
    return sources


# ============================================================
#                    HISTORIQUE DE CONVERSATION
# ============================================================

def save_conversation(query, response, sources, elapsed, lang):
    """Sauvegarde la conversation dans un fichier log"""
    log_path = os.path.join(BASE_DIR, "conversation_log.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Date : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Langue : {lang}\n")
        f.write(f"Question : {query}\n")
        f.write(f"Reponse : {response}\n")
        f.write(f"Sources : {', '.join(sources)}\n")
        f.write(f"Temps : {elapsed:.1f}s\n")


# ============================================================
#                    POINT D'ENTREE PRINCIPAL
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("  Agent RAG TGR - Initialisation")
    print("=" * 60)

    # Chargement des composants
    embeddings = load_embeddings()
    database = load_database(embeddings)
    model = load_llm()

    # Interface chat
    print("\n" + "=" * 60)
    print("  Agent RAG TGR - Pret !")
    print("  Posez vos questions en francais ou en arabe")
    print("  Commandes : 'quit' = quitter | 'clear' = effacer")
    print("=" * 60)

    while True:
        try:
            query = input("\nVous : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAu revoir !")
            break

        # Commandes speciales
        if not query:
            continue
        if query.lower() in ["quit", "exit", "q"]:
            print("\nAu revoir !")
            break
        if query.lower() == "clear":
            os.system('cls' if os.name == 'nt' else 'clear')
            continue

        # Detecter la langue
        lang = detect_language(query)
        print(f"\n[Langue detectee: {'Arabe' if lang == 'ar' else 'Francais'}]")
        print("Recherche en cours...")
        
        start = time.time()

        try:
            # Etape 1 : Recherche dans ChromaDB
            docs = search_documents(database, query)

            if not docs:
                print("\nAucun document pertinent trouve.")
                continue

            # Etape 2 : Construction du prompt
            prompt = build_prompt(query, docs)

            # Etape 3 : Generation de la reponse
            response = generate_response(model, prompt)
            elapsed = time.time() - start

            # Affichage de la reponse
            print(f"\nTGR Agent :")
            print("-" * 40)
            print(response)
            print("-" * 40)

            # Affichage des sources
            sources = show_sources(docs)
            if sources:
                print(f"\nSources ({len(sources)}) :")
                for s in sources:
                    print(s)

            print(f"\nTemps de reponse : {elapsed:.1f}s")

            # Sauvegarder la conversation
            save_conversation(query, response, sources, elapsed, lang)

        except Exception as e:
            print(f"\nErreur : {e}")


if __name__ == "__main__":
    main()