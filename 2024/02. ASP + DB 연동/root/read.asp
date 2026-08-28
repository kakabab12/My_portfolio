<% 

DSNless="DRIVER={Microsoft Access Driver (*.mdb)}; "
DSNless=DSNless & "DBQ=" & server.mappath("nwind.mdb")

Set Conn = Server.CreateObject("ADODB.Connection")
Conn.Open DSNless

Set Rs = Server.CreateObject("ADODB.Recordset")
Rs.Open "Select * From tblCategories", Conn

%>

<html>
<head>
  <meta charset="UTF-8">
</head>

<body>

<table border="2">

  <tr>
   <td>레이블1</td>
   <td>레이블2</td>
   <td>레이블3</td>
  </tr>

<%
while not Rs.eof
%>

  <tr>
    <td> 
      <input type="text" name="CID" value="<% = Rs("CategoryID")%>" > 
    </td>
    <td> 
      <input type="text" name="CName" value="<% = Rs("CategoryName")%>" > 
    </td>
    <td> 
      <input type="text" name="CDesc" value="<% = Rs("Description")%>" > 
    </td>
  </tr>

<%
Rs.movenext
Wend
%>

</table>

</body>
</html>

<%

Rs.close
set Rs=nothing
Conn.close
Set Conn=nothing

%>


