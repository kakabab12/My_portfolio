<html>
<head>
<title>Contents</title>
</head>

<body bgcolor="#FFFFFF">

<p><font size="5">Info Sys: Online Employee Directory Management</font></p>

<hr size="1" align="left" color="#000000" width="80%">
<% 
X1 = CInt(Request.Form("txtid"))
DSNless="DRIVER={Microsoft Access Driver (*.mdb)}; "
DSNless=DSNless & "DBQ=" & server.mappath("nwind.mdb")

Set Conn = Server.CreateObject("ADODB.Connection")
Conn.Open DSNless

Set Rs = Server.CreateObject("ADODB.Recordset")
Rs.Open "Select * From tblEmployees Where EmployeeID = "& X1 &";", Conn

%>
<form method="post" action="update3.asp">
<table border=1 width="492" height="155">


    <tr>
      <td width="108" height="37">Employee ID</td>
      <td width="237" height="37"><%=Rs("EmployeeID")%></td>
    </tr> 


    <tr>
      <td width="108" height="37">Last Name</td>
      <td width="237" height="37"><input type="text" name="txtLN" size="20" value="<%=Rs("LastName")%>"></td>
    </tr>
    <tr>
      <td width="108" height="37">First Name</td>
      <td width="237" height="37">
      <input type="text" name="txtFN" size="20" value="<%=Rs("FirstName")%>"></td>
    </tr>
    <tr>
      <td width="108" height="37">Title</td>
      <td width="237" height="37">
      <input type="text" name="txtT" size="20" value="<%=Rs("Title")%>"></td>
    </tr>
    <tr>
      <td width="108" height="37">Title Of Courtesy</td>
      <td width="237" height="37">
      <input type="text" name="txtTOC" size="20" value="<%=Rs("TitleOfCourtesy")%>"></td>
    </tr>


    <tr>
      <td width="108" height="37">Birth Date</td>
      <td width="237" height="37">
      <input type="text" name="txtBD" size="20" value="<%=Rs("BirthDate")%>"></td>
    </tr>


    <tr>
      <td width="108" height="38"><input type="hidden" name="txtID" value="<%=X1%>"></td>
      <td width="237" height="38"><input type="submit" value="Update" name="B2"></td>
    </tr>

</table>
</form>
<% 
Rs.close
set Rs=nothing
Conn.close
Set Conn=nothing
%>

<hr size="1" align="left" color="#000000" width="80%">

</body>
</html>