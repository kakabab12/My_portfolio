<%
X1=Request.Form("txtcid")

DSNless="DRIVER={Microsoft Access Driver (*.mdb)}; "
DSNless=DSNless & "DBQ=" & server.mappath("nwind.mdb")

Set Conn = Server.CreateObject("ADODB.Connection")
Conn.Open DSNless

sql = "DELETE * FROM tblCategories WHERE CategoryID = '"& X1 &"';"	
Conn.execute(sql)
Conn.Close

Response.Redirect "delete2.asp"
%>
