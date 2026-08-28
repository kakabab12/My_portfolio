    <HTML>
    <HEAD>
    <TITLE>Getting Employee Info</TITLE>
    </HEAD>
    <BODY>
    <H1>Employee Info</H1>
    <FORM method="post" action="GetEmpInfo.asp">
    Enter the last name of the employee you want To search:
    <BR><INPUT type=text name="empLastName" size=40 value=<%=Request("empLastName")%> > 
    <INPUT type=submit style="width=150" value="Show">
    </FORM>
    <HR>
    <span>Sample names are: <I>Davolio, Fuller, Buchanan</I></span>
    <HR><BR>
    <% 
    if Request("empLastName") = "" Then Response.End
    %>
    <% 
    
    DSNless="DRIVER={Microsoft Access Driver (*.mdb)}; "
	DSNless=DSNless & "DBQ=" & server.mappath("nwind.mdb")
    
    sql1 = "select firstname, lastname, city, hiredate, photo from tblEmployees where "
    sql2 = "lastname='"
    sql3 = Request.Form("empLastName") & "'"
    sql = sql1 & sql2 & sql3
    Set oRS = Server.CreateObject("ADODB.Recordset")
    oRS.CursorLocation = 3
    oRS.Open sql, DSNless
    SetImageForDisplay oRS("photo"), "ole"
    'oRS.Open "select logo from pub_info whe
    '     re pub_id='0736'", "DSN=PUBS;UID=sa" 
    'SetImageForDisplay oRS("logo"), "gif"
    Set oRS.ActiveConnection = Nothing
    %>
    <TABLE>
    <TR>
    <TD valign=top><B>Employee:</B><BR>
    <%=oRS("firstName") %> <%=oRS("lastName") %><BR>
    <B>from </B> <%=oRS("city") %><BR>
    <B>hired </B> <%=oRS("hiredate") %><BR>
    </TD>
    <TD>
    <Img src="theImg.asp" </Img>
    </TD>
    </TR>
    </TABLE>
    <%
    function SetImageForDisplay(field, contentType)
    OLEHEADERSIZE = 78
    
    contentType = LCase(contentType)
    Select Case contentType
      Case "gif", "jpeg", "bmp"
        contentType = "image/" & contentType
        bytes = field.value 
      Case "ole"
        contentType = "image/bmp" 
        nFieldSize = field.ActualSize
        oleHeader = field.GetChunk(OLEHEADERSIZE)
        bytes = field.GetChunk(nFieldSize - OLEHEADERSIZE)
    End Select
    Session("ImageBytes") = bytes
    Session("ImageType") = contentType
    End function
    %>
    </BODY>
    </HTML>