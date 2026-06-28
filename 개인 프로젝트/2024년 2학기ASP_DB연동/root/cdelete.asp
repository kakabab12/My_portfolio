<%
X = CInt(Request.Form("txtid"))

DSNless="DRIVER={Microsoft Access Driver (*.mdb)}; "
DSNless=DSNless & "DBQ=" & server.mappath("onchat.mdb")

Set Conn = Server.CreateObject("ADODB.Connection")
Conn.Open DSNless

sql = "DELETE * FROM IdeaTime WHERE id = "& X &";"	
Conn.execute(sql)
Conn.Close

Response.Redirect "cread.asp"
%>
