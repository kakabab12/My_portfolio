<% 

DSNless="DRIVER={Microsoft Access Driver (*.mdb)}; "
DSNless=DSNless & "DBQ=" & server.mappath("nwind.mdb")

Set Conn = Server.CreateObject("ADODB.Connection")
Conn.Open DSNless

Set Rs = Server.CreateObject("ADODB.Recordset")
Rs.Open "Select * From tblEmployees", Conn

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
     <form method="post" action="update2.asp">
      <input type="hidden" name="CID" value="<% = Rs("EmployeeID")%>" > 
      <input type="submit" value="  <% = Rs("EmployeeID")%>  ">
     </form>
    </td>
    <td> 
      <% = Rs("LastName")%>
    </td>
    <td> 
      <% = Rs("FirstName")%>
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


