<%

DSNless="DRIVER={Microsoft Access Driver (*.mdb)}; "
DSNless=DSNless & "DBQ=" & server.mappath("nwind.mdb")

Set Conn = Server.CreateObject("ADODB.Connection")
Conn.Open DSNless

sql = "UPDATE tblEmployees SET LastName = '" & Request.Form("txtLN") & "', FirstName = '" & Request.Form("txtFN") & "', Title = '" & Request.Form("txtT") & "', TitleOfCourtesy = '" & Request.Form("txtTOC") & "', BirthDate = '" & Request.Form("txtBD") & "' WHERE EmployeeID = " & CInt(Request.Form("txtID")) & ";"	
Conn.execute(sql)
Conn.Close
%>
<html>

<head>
<title>Delete</title>
<meta name="GENERATOR" content="Microsoft FrontPage 5.0">
</head>

<body>

<p align="center">&nbsp;</p>

<p align="center"><big>A record is successfully updated.</big></p>

<hr size="1" align="center" width="80%">

</body>
</html>