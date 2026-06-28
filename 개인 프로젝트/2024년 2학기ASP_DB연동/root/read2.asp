<% 

DSNless="DRIVER={Microsoft Access Driver (*.mdb)}; "
DSNless=DSNless & "DBQ=" & server.mappath("nwind.mdb")

Set Conn = Server.CreateObject("ADODB.Connection")
Conn.Open DSNless

Set Rs = Server.CreateObject("ADODB.Recordset")
Rs.Open "Select * From tblProducts;", Conn

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
    <td> <% = Rs("ProductID") %> </td>
    <td> <% = Rs("ProductName") %> </td>
    <td> <% = Rs("QuantityPerUnit") %> </td>
    <td> <% = FormatCurrency(Rs("UnitPrice"), 2) %> </td>
    <td> <% = Rs("UnitsInStock") %> </td>
    <td> <% = Rs("UnitsOnOrder") %> </td>
    <td> <% = Rs("ReOrderLevel") %> </td>
    <td> <% = Rs("Discontinued") %> </td>
    <td> <input name="Checkbox1" type="checkbox" <% If Rs("Discontinued") = True Then %> checked <% End If %> > </td>
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


