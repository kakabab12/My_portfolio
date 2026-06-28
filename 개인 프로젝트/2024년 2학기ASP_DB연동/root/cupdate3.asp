<%

DSNless="DRIVER={Microsoft Access Driver (*.mdb)}; "
DSNless=DSNless & "DBQ=" & server.mappath("onchat.mdb")

Set Conn = Server.CreateObject("ADODB.Connection")
Conn.Open DSNless

sql = "UPDATE IdeaTime SET Idea = '"& Request.Form("txtIdea") &"' WHERE id = "& CInt(Request.Form("txtid")) &";"	
Conn.execute(sql)
Conn.Close

Response.Redirect "cread.asp"
%>
