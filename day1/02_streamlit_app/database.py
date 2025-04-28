# database.py
import sqlite3
import pandas as pd
from datetime import datetime
import streamlit as st
from config import DB_FILE
from metrics import calculate_metrics # metricsを計算するために必要

class Database:
    def __init__(self, db_file):
        self.db_file = db_file
        self.init_db()

    def init_db(self):
        """データベースとテーブルを初期化する"""
        try:
            conn = sqlite3.connect(self.db_file)
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS chat_history
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 timestamp TEXT,
                 user_input TEXT,
                 model_response TEXT,
                 feedback INTEGER)
            ''')
            conn.commit()
            conn.close()
            print(f"Database '{self.db_file}' initialized successfully.")
        except Exception as e:
            st.error(f"データベースの初期化に失敗しました: {e}")
            raise e # エラーを再発生させてアプリの起動を止めるか、適切に処理する

    def add_chat(self, user_input, model_response):
        """チャット履歴をデータベースに保存する"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
            c = conn.cursor()
            timestamp = datetime.now().isoformat()

            c.execute('''
                INSERT INTO chat_history (timestamp, user_input, model_response)
                VALUES (?, ?, ?)
            ''', (timestamp, user_input, model_response))
            conn.commit()
            print("Data saved to DB successfully.") # デバッグ用
        except sqlite3.Error as e:
            st.error(f"データベースへの保存中にエラーが発生しました: {e}")
        finally:
            if conn:
                conn.close()

    def add_feedback(self, chat_id, feedback):
        """フィードバックをデータベースに保存する"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
            c = conn.cursor()
            c.execute('''
                UPDATE chat_history
                SET feedback = ?
                WHERE id = ?
            ''', (feedback, chat_id))
            conn.commit()
            print("Feedback saved to DB successfully.") # デバッグ用
        except sqlite3.Error as e:
            st.error(f"データベースへのフィードバックの保存中にエラーが発生しました: {e}")
        finally:
            if conn:
                conn.close()

    def get_chat_history(self, limit=10):
        """データベースから全てのチャット履歴を取得する"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
            c = conn.cursor()
            c.execute('''
                SELECT id, timestamp, user_input, model_response, feedback
                FROM chat_history
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            history = c.fetchall()
            return history
        except sqlite3.Error as e:
            st.error(f"履歴の取得中にエラーが発生しました: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def get_feedback_stats(self):
        """フィードバックの統計情報を取得する"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
            c = conn.cursor()
            c.execute('''
                SELECT feedback, COUNT(*) as count
                FROM chat_history
                WHERE feedback IS NOT NULL
                GROUP BY feedback
            ''')
            stats = c.fetchall()
            return stats
        except sqlite3.Error as e:
            st.error(f"フィードバックの統計情報の取得中にエラーが発生しました: {e}")
            return []
        finally:
            if conn:
                conn.close()

    def get_db_count(self):
        """データベース内のレコード数を取得する"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_file)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM chat_history")
            count = c.fetchone()[0]
            return count
        except sqlite3.Error as e:
            st.error(f"レコード数の取得中にエラーが発生しました: {e}")
            return 0
        finally:
            if conn:
                conn.close()

    def clear_db(self):
        """データベースの全レコードを削除する"""
        conn = None
        confirmed = st.session_state.get("confirm_clear", False)

        if not confirmed:
            st.warning("本当にすべてのデータを削除しますか？もう一度「データベースをクリア」ボタンを押すと削除が実行されます。")
            st.session_state.confirm_clear = True
            return False # 削除は実行されなかった

        try:
            conn = sqlite3.connect(self.db_file)
            c = conn.cursor()
            c.execute("DELETE FROM chat_history")
            conn.commit()
            st.success("データベースが正常にクリアされました。")
            st.session_state.confirm_clear = False # 確認状態をリセット
            return True # 削除成功
        except sqlite3.Error as e:
            st.error(f"データベースのクリア中にエラーが発生しました: {e}")
            st.session_state.confirm_clear = False # エラー時もリセット
            return False # 削除失敗
        finally:
            if conn:
                conn.close()