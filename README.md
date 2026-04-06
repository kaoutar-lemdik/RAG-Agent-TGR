Agent RAG — Trésorerie Générale du Royaume

## Un assistant intelligent bilingue (Français/Arabe) pour les documents de la TGR

---

##  Table des matières

1. [C'est quoi ce projet ?]
2. [Le problème]
3. [La solution]
4. [Comment ça marche ?]
5. [Démonstration]
6. [Technologies utilisées]
7. [Architecture du projet]
8. [Les chiffres clés]
9. [Installation & Lancement]
10. [Structure du projet]
11. [Défis rencontrés]
12. [Auteur]

---

##  C'est quoi ce projet ?

Imaginez que vous avez **des centaines de documents** (PDF, DOCX) 
de la Trésorerie Générale du Royaume (TGR) : des lois, des décrets, 
des circulaires, des notes...

**Trouver une information précise** dans tout ça peut prendre 
des heures de recherche manuelle.

 **Cet agent résout ce problème.**

C'est un **assistant intelligent** à qui vous pouvez poser des questions 
en **français** ou en **arabe**, et il vous donne la réponse exacte 
en s'appuyant sur les documents officiels.

>  Exemple : 
> Vous demandez : *"Quel est le délai de paiement des marchés publics ?"*
> 
>  L'agent répond avec le décret exact et les montants précis, 
> avec la source du document.

---

##  Le Problème

 Avant (sans l'agent) :

Employé : "Je dois trouver les conditions de nantissement
des marchés publics..."
↓
 Ouvrir des dizaines de fichiers PDF
↓
 Chercher manuellement dans chaque page
↓
 Passer des heures à trouver l'information
↓
 Risque de manquer l'information importante

## La Solution

 Après (avec l'agent) :

Employé : "ما هي شروط رهن الصفقات العمومية؟"
↓
 L'agent comprend la question (français OU arabe)
↓
 Recherche automatique dans TOUS les documents
↓
 Trouve les passages les plus pertinents
↓
 Génère une réponse claire et précise
↓
 Cite les sources (quel document, quelle page)
↓
 Réponse en quelques secondes !

## ⚙️ Comment ça marche ?

### Explication simple (pour tout le monde) :

 L'agent fonctionne en 3 grandes étapes :

┌─────────────────────────────────────────────────┐
│ │
│ ÉTAPE 1 : APPRENDRE  │
│ │
│ L'agent "lit" tous les documents PDF et DOCX │
│ Il les découpe en petits morceaux │
│ Il les mémorise dans une base de données │
│ │
├─────────────────────────────────────────────────┤
│ │
│ ÉTAPE 2 : COMPRENDRE  │
│ │
│ Quand vous posez une question, l'agent : │
│ → Détecte la langue (français ou arabe) │
│ → Comprend le sens de votre question │
│ → Cherche les morceaux les plus pertinents │
│ │
├─────────────────────────────────────────────────┤
│ │
│ ÉTAPE 3 : RÉPONDRE  │
│ │
│ L'agent utilise l'IA pour : │
│ → Formuler une réponse claire │
│ → S'appuyer sur les documents trouvés │
│ → Citer les sources │
│ │
└─────────────────────────────────────────────────┘

### Explication technique (pour les développeurs) :

 Documents (PDF/DOCX)
│
▼
┌──────────────────┐
│ 1. CHARGEMENT │ PyPDFLoader / Docx2txtLoader
│ & NETTOYAGE │ Regex (FR + AR)
└────────┬─────────┘
│
▼
┌──────────────────┐
│ 2. DÉCOUPAGE │ RecursiveCharacterTextSplitter
│ (Chunking) │ 500 chars | overlap 50
│ │ Séparateurs FR + AR (، ؟ !)
└────────┬─────────┘
│
▼
┌──────────────────┐
│ 3. EMBEDDINGS │ HuggingFace multilingual-e5-base
│ (Encodage) │ Préfixe "passage:" pour E5
└────────┬─────────┘
│
▼
┌──────────────────┐
│ 4. STOCKAGE │ ChromaDB (base vectorielle)
│ │ Batch de 500 chunks
└────────┬─────────┘
│
▼
┌──────────────────┐
│ 5. RECHERCHE │ Similarity Search
│ SÉMANTIQUE │ Préfixe "query:" pour E5
└────────┬─────────┘
│
▼
┌──────────────────┐
│ 6. GÉNÉRATION │ Llama-3-8B (LLM)
│ DE RÉPONSE │ Prompt bilingue FR/AR
└──────────────────┘

##  Démonstration

### 🇲🇦 Exemple en Arabe :

> **Question :** ما هي شروط رهن الصفقات العمومية؟
> 
> **Réponse :** وفقاً لما هو وارد في المادة 3 من القانون، 
> شروط رهن الصفقات العمومية هي:
> 1. أن يكون عقد الرهن متافقاً ومقبولاً من طرف صاحب الصفقة...
> 2. أن يتضمن العقد كل البيانات الضرورية لتنفيذه...
> 
>  **Source :** Document X, Page Y

### 🇫🇷 Exemple en Français :

> **Question :** Quel est le délai de paiement des marchés publics ?
> 
> **Réponse :** Selon le décret n° 2-12-349 du 8 joumada I 1434 
> (20 mars 2013) relatif aux marchés publics, les délais de 
> paiement sont fixés à :
> • 120 jours pour les marchés ≥ 2 millions de dirhams
> • 105 jours pour les marchés ≥ 1 million de dirhams
> 
>  **Source :** Décret n° 2-12-349

---

##  Technologies utilisées

### Pour les non-techniques :

| Technologie | Rôle (en simple) |
|---|---|
|  **Python** | Le langage de programmation utilisé |
|  **Llama-3-8B** | Le "cerveau" qui génère les réponses |
|  **E5-multilingual** | Le traducteur qui comprend FR et AR |
|  **ChromaDB** | La mémoire où sont stockés les documents |
|  **LangChain** | Le chef d'orchestre qui coordonne tout |
|  **Streamlit** | L'interface web (ce que l'utilisateur voit) |

### Pour les techniques :

| Composant | Technologie | Détails |
|---|---|---|
| **LLM** | Llama-3-8B | Via llama.cpp, quantifié, CPU |
| **Embeddings** | intfloat/multilingual-e5-base | HuggingFace, normalize=True |
| **Vector Store** | ChromaDB | Persistant, SQLite backend |
| **Framework** | LangChain | Loaders, Splitters, Chains |
| **Text Splitting** | RecursiveCharacterTextSplitter | 500 chars, overlap 50 |
| **Frontend** | Streamlit | Interface conversationnelle |
| **Loaders** | PyPDFLoader, Docx2txtLoader | PDF + DOCX support |

---

##  Les chiffres clés

┌────────────────────────────────────────┐
│ │
│ 📁 50 fichiers sources (PDF + DOCX) │
│ │
│ 📄 943 pages traitées │
│ │
│ ✂️ 5 534 chunks vectorisés │
│ │
│ 🌐 2 langues supportées (FR + AR) │
│ │
│ ⏱️ ~26 minutes pour l'ingestion │
│ │
│ 💬 Réponses en ~3-4 minutes (CPU) │
│ │
└────────────────────────────────────────┘

##  Installation & Lancement

Prérequis :

- Python 3.10 ou supérieur
- 8 Go de RAM minimum
- Espace disque : ~10 Go (pour le modèle Llama-3)

Étape 1 : Cloner le projet
```bash
git clone https://github.com/votre-username/RAG-TGR.git
cd RAG-TGR


Étape 2 : Installer les dépendances

pip install -r requirements.txt

Étape 3 : Placer vos documents

Data_full/

Étape 4 : Lancer l'ingestion des documents

python ingestion_chroma.py
text
# Résultat attendu :
# ✅ INGESTION TERMINÉE AVEC SUCCÈS
# Documents traités : 943
# Chunks stockés    : 5534
# Temps total       : 1574.4s

Étape 5 : Lancer l'application

streamlit run app.py
text
# L'application s'ouvre dans votre navigateur :
# 🌐 http://localhost:8501

📁 Structure du projet


RAG_TGR/
│
├── 📂 Chroma_db/              # Base de données vectorielle
│   └── chroma.sqlite3         # Stockage des embeddings
│
├── 📂 Data_full/              # Documents complets (PDF + DOCX)
│   ├── document1.pdf
│   ├── document2.docx
│   └── ...
│
├── 📂 Data_test/              # Documents de test
│   └── ...
│
├── 📂 Models/                 # Modèle LLM (Llama-3-8B)
│   └── llama-3-8b.gguf
│
├── 🐍 app.py                  # Application Streamlit (interface)
├── 🐍 chat_rag.py             # Logique du chatbot RAG
├── 🐍 ingestion_chroma.py     # Pipeline d'ingestion des documents
├── 📄 conversation_log        # Historique des conversations
├── 📄 requirements.txt        # Dépendances Python
└── 📄 README.md               # Ce fichier


Défis rencontrés

1.  Gestion bilingue (Français + Arabe)

Problème : Les documents contiennent du texte en français 
           ET en arabe, parfois mélangés.

Solution : → Utilisation d'un modèle d'embeddings MULTILINGUE
             (multilingual-e5-base)
           → Séparateurs de texte adaptés à l'arabe (، ؟ !)
           → Nettoyage spécifique des caractères arabes

2.  Contraintes matérielles (CPU only)

Problème : GPU disponible (Quadro P520) avec seulement 
           2 Go VRAM → insuffisant pour le LLM.

Solution : → Exécution 100% sur CPU
           → Modèle Llama-3-8B quantifié (GGUF)
           → Batch processing pour l'ingestion (lots de 500)

3.  Qualité des documents

Problème : Les PDF administratifs contiennent souvent des 
           caractères parasites, des encodages cassés, etc.

Solution : → Nettoyage regex avancé (clean_text)
           → Suppression des caractères de contrôle
           → Filtrage des chunks trop courts (< 20 chars)

Auteur
Lemdik Kaoutar

🎓 Étudiante en Ingénierie de la Décision
🏫 Université Sultan Moulay Slimane
📍 Sale, Maroc

🔗 https://www.linkedin.com/in/lemdik-kaoutar
📧 kaoutar.lemdik@um5r.ac.ma

Remerciements
La Trésorerie Générale du Royaume — Pour l'opportunité de stage
La communauté open source — LangChain, HuggingFace, ChromaDB, Meta (Llama-3)


