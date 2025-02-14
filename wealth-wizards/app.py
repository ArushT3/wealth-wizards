import os
from traceback import print_list

from cs50 import SQL
from flask import Flask, redirect, render_template, request, session
from sqlalchemy.orm.attributes import backref_listeners
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from flask_session import Session
from helpers import apology
from re import fullmatch

#Loads the database and initialises the session
app = Flask(__name__)
db = SQL("sqlite:///wealth_wizards.db")


app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)



#Login Function
@app.route("/login", methods=["GET", "POST"])


def login():
    session.clear()
    if request.method == "GET":
        return render_template("login.html")
    else:
        if not request.form.get("username"):
            return apology("must provide username", "/login", "Go back to login")

        elif not request.form.get("password"):
            return apology("must provide password", "/login", "Go back to login")

        rows = db.execute("SELECT * FROM users WHERE username = ?", request.form.get("username"))

        if len(rows) != 1 or not check_password_hash(rows[0]["hash"], request.form.get("password")):
            return apology("invalid username and/or password", "/login", redirect_text="Go back to login")

        session["user_id"] = rows[0]["id"]

        return redirect("/")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""
    session.clear()
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        confirmation = request.form.get("confirmation")

        rows_u = db.execute("SELECT * FROM users WHERE username = ?", username)
        if not username:
            return apology("Blank Username", "/register", "Go back to register")
        elif not password:
            return apology("Blank Password", "/register", "Go back to register")
        elif len(rows_u) != 0:
            return apology("Username Exists", "/register", "Go back to register")
        elif not confirmation:
            return apology("Blank Confirmation", "/register", "Go back to register")
        elif password != confirmation:
            return apology("Password not matching confirmation", "/register", "Go back to register")
        else:
            db.execute("INSERT INTO users (username, hash) VALUES(?, ?)", username,
                       generate_password_hash(password))
        return redirect("/login")
    else:
        return render_template("register.html")


@app.route("/")
def index():
    if not session.get("user_id"):
        return redirect("/login")

    name = db.execute("SELECT username FROM users WHERE id=?", session["user_id"])
    p_list =  []
    return render_template("index.html", name=name[0]["username"], list=p_list)


@app.route("/addstock", methods=["GET", "POST"])
def addstock():
    if request.method == "POST":
        stock = request.form.get("stock")
        type = request.form.get("type")
        if not stock:
            return apology("Please enter Stock", "/addstock", "Go back to Add a Stock")
        if not type:
            return apology("Please enter Type", "/addstock", "Go back to Add a Stock")
        db.execute(
            "INSERT INTO stocks (stock, type, user_id) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            stock, type, session["user_id"])
        return redirect("/")
    else:
        return render_template("addstock.html")


@app.route("/stocks", methods=["GET", "POST"])
def stocks():
    return render_template("stocks.html", options=[])


@app.route("/backtest", methods=["GET", "POST"])
def backtest():
    if request.method == "POST":
        stock = request.form.get("stock")
        type = request.form.get("type")
        if not stock:
            return apology("Please enter Stock", "/backtest", "Go back to backtest")
        if not type:
            return apology("Please enter Type", "/backtest", "Go back to backtest")
        p_list = []
        r_list = []
        b_photo = "./b.png"
        return render_template("results.html", message="Backtest Results for" + stock, list=p_list, results_list=r_list, backtest_p=b_photo)
    else:
        return render_template("backtest.html")


@app.route("/terms")
def terms():
    return render_template("terms.html")