<%

X = Request.Form("txtIdea")

DSNless="DRIVER={Microsoft Access Driver (*.mdb)}; "
DSNless=DSNless & "DBQ=" & server.mappath("onchat.mdb")

Set Conn = Server.CreateObject("ADODB.Connection")
Conn.Open DSNless

SQL = "INSERT INTO IdeaTime (Idea) VALUES ('"& X &"')"
Conn.Execute SQL

Conn.Close

Set Conn=nothing

Response.Redirect "cwrite.htm"
%>