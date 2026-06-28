<%

X1 = Request.Form("txtcid")
X2 = Request.Form("txtcname")
X3 = Request.Form("txtdesc")

DSNless="DRIVER={Microsoft Access Driver (*.mdb)}; "
DSNless=DSNless & "DBQ=" & server.mappath("nwind.mdb")

Set Conn = Server.CreateObject("ADODB.Connection")
Conn.Open DSNless

SQL = "INSERT INTO tblCategories (CategoryID, CategoryName, Description) VALUES ('"&X1&"','"&X2&"','"&X3&"')"
Conn.Execute SQL

'SQL = "SELECT * FROM tblCategories"
'Set rs = Server.CreateObject("adodb.recordset")
'rs.Open SQL, Conn
'rs.AddNew
'rs("CategoryID")= X1
'rs("CategoryName") = X2
'rs("Description")= X3
'rs.Update
'rs.Close

Conn.Close

Set Conn=nothing

Response.Redirect "insert.htm"
%>