<%

X1 = Request.Form("txtpid")
X2 = CInt(Request.Form("txtup"))
X3 = CInt(Request.Form("txtuis"))
X4 = CInt(Request.Form("txtuoo"))
X5 = CInt(Request.Form("txtrl"))

If Request.Form("txtd") = "Yes" Then
   X6 = TRUE
End If

If Request.Form("txtd") = "No" Then
   X6 = FALSE
End If

DSNless="DRIVER={Microsoft Access Driver (*.mdb)}; "
DSNless=DSNless & "DBQ=" & server.mappath("nwind.mdb")

Set Conn = Server.CreateObject("ADODB.Connection")
Conn.Open DSNless

SQL = "INSERT INTO tblProducts (ProductID, UnitPrice, UnitsInStock, UnitsOnOrder, ReOrderLevel) VALUES ('"&X1&"',"&X2&", "&X3&", "&X4&", "&X5&", "&X6&")"
Conn.Execute SQL

Conn.Close

Set Conn=nothing

Response.Redirect "insert2.htm"
%>