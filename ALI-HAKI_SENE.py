import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import hashlib
import json
import os
from datetime import datetime
import re
import uuid
import warnings

# Configuration des warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="coroutine 'expire_cache' was never awaited")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# Configuration de la page
st.set_page_config(
    page_title="Movie Recommender Pro",
    page_icon="🎬",
    layout="wide"
)

# ---- CONSTANTS ----
DATA_PATH = "data/user_ratings_genres_mov.csv"
USERS_DIR = "data/users"
USER_RATINGS_PATH = "data/user_ratings_updated.csv"  # Nouveau fichier pour les données combinées
os.makedirs(USERS_DIR, exist_ok=True)

# ---- STYLES ----
def load_css():
    st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #2e3b4e, #1e293b);
            color: white;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1rem;
        }
        .stTextInput>div>div>input {
            border: 2px solid #4CAF50 !important;
            border-radius: 8px;
        }
        .movie-card {
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            background-color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .profile-header {
            display: flex;
            align-items: center;
            margin-bottom: 2rem;
        }
        .profile-avatar {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            object-fit: cover;
            margin-right: 1.5rem;
        }
        .recommendation-type {
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #4CAF50;
        }
    </style>
    """, unsafe_allow_html=True)

load_css()

# ---- AUTHENTICATION ----
def hash_password(password: str) -> str:
    """Hash un mot de passe avec SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate(email: str, password: str) -> bool:
    """Authentifie un utilisateur"""
    user = load_user(email)
    return user and user["password"] == hash_password(password)

def change_password(email: str, old_password: str, new_password: str) -> bool:
    """Change le mot de passe d'un utilisateur"""
    user = load_user(email)
    if not user or user["password"] != hash_password(old_password):
        return False
    
    user["password"] = hash_password(new_password)
    save_user(email, user)
    return True

# ---- UTILITY FUNCTIONS ----
def is_valid_email(email):
    """Valide le format d'un email"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def generate_user_id():
    """Génère un ID utilisateur unique"""
    return str(uuid.uuid4())

# ---- DATA MANAGEMENT ----
@st.cache_data
def load_movie_data():
    """Charge les données des films"""
    try:
        df = pd.read_csv(DATA_PATH)
        
        # Nettoyage et préparation des données
        if not {'userId', 'title', 'genres', 'rating'}.issubset(df.columns):
            st.error("Le fichier CSV ne contient pas les colonnes requises")
            return pd.DataFrame()
            
        df = df.rename(columns={'userId': 'user_id', 'title': 'movie_title'})
        df['movie_id'] = df.groupby('movie_title').ngroup()
        
        return df
    except Exception as e:
        st.error(f"Erreur de chargement : {str(e)}")
        return pd.DataFrame()

def save_user(email: str, user_data: dict):
    """Sauvegarde les données utilisateur"""
    user_data['last_updated'] = datetime.now().isoformat()
    with open(f"{USERS_DIR}/{email}.json", "w") as f:
        json.dump(user_data, f)

def load_user(email: str):
    """Charge les données utilisateur"""
    try:
        with open(f"{USERS_DIR}/{email}.json", "r") as f:
            return json.load(f)
    except:
        return None

def integrate_user_ratings(df: pd.DataFrame, user_ratings: dict, user_id: str):
    """
    Intègre les évaluations utilisateur dans le dataset principal
    Retourne un nouveau DataFrame avec les évaluations intégrées
    """
    if not user_ratings:
        return df
    
    # Création des nouvelles lignes pour les évaluations utilisateur
    new_ratings = []
    for movie_title, rating in user_ratings.items():
        movie_data = df[df['movie_title'] == movie_title].iloc[0].copy()
        movie_data['user_id'] = user_id
        movie_data['rating'] = rating
        new_ratings.append(movie_data)
    
    # Concaténation avec le dataset original
    new_df = pd.concat([df, pd.DataFrame(new_ratings)], ignore_index=True)
    
    # Sauvegarde du nouveau dataset
    new_df.to_csv(USER_RATINGS_PATH, index=False)
    
    return new_df

# ---- RECOMMENDATION SYSTEM ----
class RecommenderSystem:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._prepare_data()
    
    def _prepare_data(self):
        """Prépare les données pour les algorithmes"""
        # Vectorisation des genres
        self.tfidf = TfidfVectorizer(tokenizer=lambda x: x.split('|'), token_pattern=None)
        self.genres_matrix = self.tfidf.fit_transform(self.df['genres'].fillna(''))
        
        # Préparation de la matrice utilisateur-item pour les méthodes collaboratives
        self.user_item_matrix = self.df.pivot_table(
            index='user_id', 
            columns='movie_title', 
            values='rating'
        ).fillna(0)
    
    def content_based_recommendations(self, movie_title: str, n=5) -> pd.DataFrame:
        """Recommandations basées sur les genres sans doublons"""
        try:
            # Trouver l'index du film de référence
            idx = self.df[self.df['movie_title'] == movie_title].index[0]
            
            # Calculer les similarités
            sim_scores = cosine_similarity(self.genres_matrix[idx], self.genres_matrix).flatten()
            
            # Trier par score de similarité (du plus élevé au plus bas)
            similar_indices = np.argsort(-sim_scores)
            
            # Préparer les résultats sans doublons
            seen_titles = set()
            unique_results = []
            
            for i in similar_indices:
                current_title = self.df.iloc[i]['movie_title']
                
                # Ignorer le film de référence et les doublons
                if current_title != movie_title and current_title not in seen_titles:
                    seen_titles.add(current_title)
                    unique_results.append({
                        'movie_title': current_title,
                        'genres': self.df.iloc[i]['genres'],
                        'similarity_score': sim_scores[i]
                    })
                    
                    # Stop quand on a assez de résultats uniques
                    if len(unique_results) >= n:
                        break
                        
            return pd.DataFrame(unique_results)
        
        except Exception as e:
            st.error(f"Erreur dans les recommandations: {str(e)}")
            return pd.DataFrame()

    def collaborative_user_item(self, user_id: str, n=5) -> pd.DataFrame:
        """Recommandation collaborative User-Item"""
        try:
            # Trouver les utilisateurs similaires
            knn = NearestNeighbors(n_neighbors=5, metric='cosine')
            knn.fit(self.user_item_matrix)
            
            if user_id not in self.user_item_matrix.index:
                return pd.DataFrame()
                
            distances, indices = knn.kneighbors(
                self.user_item_matrix.loc[user_id].values.reshape(1, -1)
            )
            
            # Obtenir les films notés par les voisins mais pas par l'utilisateur
            similar_users = self.user_item_matrix.index[indices.flatten()]
            recommendations = self.user_item_matrix.loc[similar_users].mean(axis=0)
            user_rated = self.user_item_matrix.loc[user_id]
            recommendations = recommendations[user_rated == 0]  # Exclure déjà notés
            
            # Retourner les top n films
            top_movies = recommendations.sort_values(ascending=False).head(n)
            return pd.DataFrame({
                'movie_title': top_movies.index,
                'predicted_rating': top_movies.values
            })
        except:
            return pd.DataFrame()

    def collaborative_item_user(self, user_ratings: dict, n=5) -> pd.DataFrame:
        """Recommandation collaborative Item-User"""
        try:
            # Créer une matrice de similarité entre films
            item_similarity = cosine_similarity(self.user_item_matrix.T)
            item_similarity_df = pd.DataFrame(
                item_similarity,
                index=self.user_item_matrix.columns,
                columns=self.user_item_matrix.columns
            )
            
            # Calculer les prédictions
            predictions = []
            for movie in self.user_item_matrix.columns:
                if movie not in user_ratings:
                    # Ponderer les similarités par les notes de l'utilisateur
                    sim_scores = item_similarity_df[movie]
                    user_rated_movies = [m for m in user_ratings if m in sim_scores.index]
                    if not user_rated_movies:
                        continue
                        
                    weighted_sum = sum(
                        user_ratings[m] * sim_scores[m] 
                        for m in user_rated_movies
                    )
                    sum_sim = sum(
                        abs(sim_scores[m]) 
                        for m in user_rated_movies
                    )
                    if sum_sim > 0:
                        predictions.append((movie, weighted_sum / sum_sim))
            
            # Retourner les top n films
            predictions.sort(key=lambda x: x[1], reverse=True)
            return pd.DataFrame(
                predictions[:n], 
                columns=['movie_title', 'predicted_rating']
            )
        except:
            return pd.DataFrame()

    def nmf_recommendations(self, user_id: str, n=5) -> pd.DataFrame:
        """Recommandation par NMF"""
        try:
            model = NMF(n_components=10, init='random', random_state=42)
            W = model.fit_transform(self.user_item_matrix)
            H = model.components_
            
            if user_id not in self.user_item_matrix.index:
                return pd.DataFrame()
                
            user_idx = self.user_item_matrix.index.get_loc(user_id)
            user_pred = np.dot(W[user_idx], H)
            
            predictions = pd.Series(
                user_pred,
                index=self.user_item_matrix.columns
            )
            
            # Exclure les films déjà notés
            user_rated = self.user_item_matrix.loc[user_id]
            predictions = predictions[user_rated == 0]
            
            return pd.DataFrame({
                'movie_title': predictions.sort_values(ascending=False).head(n).index,
                'predicted_rating': predictions.sort_values(ascending=False).head(n).values
            })
        except:
            return pd.DataFrame()

    def svd_recommendations(self, user_id: str, n=5) -> pd.DataFrame:
        """Recommandation par SVD"""
        try:
            model = TruncatedSVD(n_components=10, random_state=42)
            reduced = model.fit_transform(self.user_item_matrix)
            
            if user_id not in self.user_item_matrix.index:
                return pd.DataFrame()
                
            user_idx = self.user_item_matrix.index.get_loc(user_id)
            user_pred = np.dot(reduced[user_idx], model.components_)
            
            predictions = pd.Series(
                user_pred,
                index=self.user_item_matrix.columns
            )
            
            # Exclure les films déjà notés
            user_rated = self.user_item_matrix.loc[user_id]
            predictions = predictions[user_rated == 0]
            
            return pd.DataFrame({
                'movie_title': predictions.sort_values(ascending=False).head(n).index,
                'predicted_rating': predictions.sort_values(ascending=False).head(n).values
            })
        except:
            return pd.DataFrame()

# ---- INTERFACE SECTIONS ----
def login_section():
    """Affiche le formulaire de connexion"""
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Mot de passe", type="password")
        
        if st.form_submit_button("Se connecter"):
            if authenticate(email, password):
                st.session_state.update({
                    'logged_in': True,
                    'current_user': email,
                    'current_page': 'profile'
                })
                st.rerun()
            else:
                st.error("Email ou mot de passe incorrect")

def register_section():
    """Affiche le formulaire d'inscription"""
    with st.form("register_form"):
        st.subheader("Informations personnelles")
        first_name = st.text_input("Prénom*")
        last_name = st.text_input("Nom*")
        email = st.text_input("Email*")
        
        st.subheader("Sécurité")
        password = st.text_input("Mot de passe*", type="password")
        confirm_password = st.text_input("Confirmez le mot de passe*", type="password")
        
        if st.form_submit_button("S'inscrire"):
            # Validation
            if not all([first_name, last_name, email, password, confirm_password]):
                st.error("Tous les champs marqués d'un * sont obligatoires")
            elif password != confirm_password:
                st.error("Les mots de passe ne correspondent pas")
            elif not is_valid_email(email):
                st.error("Veuillez entrer un email valide")
            elif load_user(email):
                st.error("Cet email est déjà utilisé")
            else:
                user_data = {
                    "first_name": first_name,
                    "last_name": last_name,
                    "email": email,
                    "password": hash_password(password),
                    "user_id": generate_user_id(),  # Ce champ est essentiel
                    "created_at": datetime.now().isoformat(),
                    "ratings": {}
                }
                save_user(email, user_data)
                st.success("Compte créé avec succès ! Vous pouvez maintenant vous connecter.")

def profile_section():
    """Gestion du profil utilisateur"""
    user = load_user(st.session_state['current_user'])
    if not user:
        st.error("Profil introuvable")
        return
    
    # Affichage du profil
    st.markdown(f"""
    <div class="profile-header">
        <img src="https://ui-avatars.com/api/?name={user['first_name']}+{user['last_name']}&background=4CAF50&color=fff&size=80" class="profile-avatar">
        <div>
            <h1>{user['first_name']} {user['last_name']}</h1>
            <p>{user['email']}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Onglets du profil
    tab1, tab2, tab3 = st.tabs(["Mon Profil", "Changer mot de passe", "Mes notes"])
    
    with tab1:
        # Modification du profil
        with st.form("edit_profile_form"):
            new_first_name = st.text_input("Prénom", value=user['first_name'])
            new_last_name = st.text_input("Nom", value=user['last_name'])
            
            if st.form_submit_button("Enregistrer les modifications"):
                user['first_name'] = new_first_name
                user['last_name'] = new_last_name
                save_user(user['email'], user)
                st.success("Profil mis à jour avec succès !")
    
    with tab2:
        # Changement de mot de passe
        with st.form("change_password_form"):
            old_password = st.text_input("Ancien mot de passe", type="password")
            new_password = st.text_input("Nouveau mot de passe", type="password")
            confirm_password = st.text_input("Confirmez le nouveau mot de passe", type="password")
            
            if st.form_submit_button("Changer le mot de passe"):
                if new_password != confirm_password:
                    st.error("Les nouveaux mots de passe ne correspondent pas")
                elif change_password(user['email'], old_password, new_password):
                    st.success("Mot de passe changé avec succès !")
                else:
                    st.error("Ancien mot de passe incorrect")
    
    with tab3:
        # Notation des films
        df = load_movie_data()
        if df.empty:
            st.warning("Aucune donnée de film disponible")
        else:
            with st.expander("Noter un nouveau film"):
                selected_movie = st.selectbox("Choisir un film", df['movie_title'].unique())
                rating = st.slider("Note", 1, 5, 3)
                
                if st.button("Enregistrer la note"):
                    if len(user['ratings']) >= 3:
                        st.error("Vous avez déjà noté 3 films (maximum autorisé)")
                    else:
                        user['ratings'][selected_movie] = rating
                        save_user(user['email'], user)
                        st.success("Note enregistrée !")
            
            # Affichage des notes existantes
            st.subheader("Mes films notés (3 maximum)")
            if user['ratings']:
                for movie, rating in user['ratings'].items():
                    cols = st.columns([0.7, 0.2, 0.1])
                    cols[0].write(movie)
                    cols[1].write("⭐" * rating)
                    if cols[2].button("✕", key=f"del_{movie}"):
                        user['ratings'].pop(movie)
                        save_user(user['email'], user)
                        st.rerun()
            else:
                st.info("Vous n'avez pas encore noté de films")

def recommendations_section():
    """Affiche les recommandations"""
    user = load_user(st.session_state['current_user'])
    
    # Vérification et génération d'un user_id si manquant
    if 'user_id' not in user:
        user['user_id'] = generate_user_id()
        save_user(user['email'], user)
    
    df = load_movie_data()
    
    if not user['ratings']:
        st.warning("Notez des films pour obtenir des recommandations")
        return
    
    if len(user['ratings']) < 3:
        st.warning(f"Veuillez noter {3 - len(user['ratings'])} film(s) supplémentaire(s) pour obtenir des recommandations")
        return
    
    # Intégration des évaluations utilisateur dans le dataset
    updated_df = integrate_user_ratings(df, user['ratings'], user['user_id'])
    recommender = RecommenderSystem(updated_df)
    best_movie = max(user['ratings'].items(), key=lambda x: x[1])[0]
    
    st.header("🍿 Recommandations pour vous")
    
    # Recommandation basée sur le contenu
    st.markdown('<div class="recommendation-type">🎭 Basées sur le contenu (genres similaires)</div>', unsafe_allow_html=True)
    content_recs = recommender.content_based_recommendations(best_movie)
    if not content_recs.empty:
        for _, row in content_recs.iterrows():
            with st.container(border=True):
                st.markdown(f"**{row['movie_title']}**")
                st.caption(f"Genres: {row['genres']}")
                st.caption(f"Score de similarité: {row['similarity_score']:.2f}")
    else:
        st.warning("Aucune recommandation basée sur le contenu disponible")
    
    # Recommandations collaboratives
    st.markdown('<div class="recommendation-type">👥 Collaboratives (User-Item)</div>', unsafe_allow_html=True)
    user_item_recs = recommender.collaborative_user_item(user['user_id'])
    if not user_item_recs.empty:
        for _, row in user_item_recs.iterrows():
            with st.container(border=True):
                st.markdown(f"**{row['movie_title']}**")
                st.caption(f"Note prédite: {row['predicted_rating']:.2f}")
    else:
        st.warning("Aucune recommandation collaborative User-Item disponible")
    
    st.markdown('<div class="recommendation-type">🎬 Collaboratives (Item-User)</div>', unsafe_allow_html=True)
    item_user_recs = recommender.collaborative_item_user(user['ratings'])
    if not item_user_recs.empty:
        for _, row in item_user_recs.iterrows():
            with st.container(border=True):
                st.markdown(f"**{row['movie_title']}**")
                st.caption(f"Note prédite: {row['predicted_rating']:.2f}")
    else:
        st.warning("Aucune recommandation collaborative Item-User disponible")
    
    st.markdown('<div class="recommendation-type">🔢 NMF (Factorisation de matrices)</div>', unsafe_allow_html=True)
    nmf_recs = recommender.nmf_recommendations(user['user_id'])
    if not nmf_recs.empty:
        for _, row in nmf_recs.iterrows():
            with st.container(border=True):
                st.markdown(f"**{row['movie_title']}**")
                st.caption(f"Note prédite: {row['predicted_rating']:.2f}")
    else:
        st.warning("Aucune recommandation NMF disponible")
    
    st.markdown('<div class="recommendation-type">📊 SVD (Décomposition en valeurs singulières)</div>', unsafe_allow_html=True)
    svd_recs = recommender.svd_recommendations(user['user_id'])
    if not svd_recs.empty:
        for _, row in svd_recs.iterrows():
            with st.container(border=True):
                st.markdown(f"**{row['movie_title']}**")
                st.caption(f"Note prédite: {row['predicted_rating']:.2f}")
    else:
        st.warning("Aucune recommandation SVD disponible")

# ---- MAIN APP ----
def main():
    """Gestion principale de l'application"""
    # Initialisation de session
    if 'logged_in' not in st.session_state:
        st.session_state.update({
            'logged_in': False,
            'current_user': None,
            'current_page': 'profile'
        })
    
    # Sidebar Navigation
    with st.sidebar:
        st.title("🎬 MovieRec Pro")
        if st.session_state['logged_in']:
            user = load_user(st.session_state['current_user'])
            if user:
                st.write(f"Connecté en tant que {user['first_name']}")
            
            # Navigation
            if st.button("👤 Mon Profil"):
                st.session_state['current_page'] = 'profile'
                st.rerun()
                
            if st.button("🍿 Recommandations"):
                st.session_state['current_page'] = 'recommendations'
                st.rerun()
                
            if st.button("🔒 Déconnexion"):
                st.session_state.update({
                    'logged_in': False,
                    'current_user': None
                })
                st.rerun()
        else:
            menu = st.radio("Menu", ["Connexion", "Inscription"], key="auth_mode")
    
    # Contenu principal
    if st.session_state['logged_in']:
        if st.session_state['current_page'] == 'profile':
            profile_section()
        else:
            recommendations_section()
    else:
        if st.session_state.get('auth_mode') == "Connexion":
            login_section()
        else:
            register_section()

if __name__ == "__main__":
    main()