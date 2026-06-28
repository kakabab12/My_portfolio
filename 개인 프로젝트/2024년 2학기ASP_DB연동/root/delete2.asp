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

<%
while not Rs.eof
%>

  <tr>
    <td> 
      <form method="post" action="delete.asp">
        <input type="hidden" name="txtcid" value="<% = Rs("CategoryID")%>">
        <input type="submit" value="<% = Rs("CategoryID")%>">
      </form>
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


