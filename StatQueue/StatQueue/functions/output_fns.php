<?php

require_once('db_fns.php');

function display_email_form(){
?>
  <h2>Send us a message</h2>
  <form method="post">
  <table width=420 bgcolor=#cccccc cellpadding="2" cellspacing=4>
  <tr><td width=10%><strong>Your Email: </strong>
<?php
  if ($user = check_valid_user()){
  	$userid = $_SESSION[$user];
  	$db = db_connect();
  	$query = "select email from authorized_users where userid='$userid'";
  	$result = $db->query($query);
  	$email = $result->fetch_row()[0];
  	echo "<input size=25 name=user_email type=text value=$email /></td></tr>";
  } else {
  	echo "<input size=25 name=user_email type=text /></td></tr>";
  }
?>
  <tr><td><strong>Message Title: </strong><input size=48 name="title" type="text" /></td></tr>
  <tr><td colspan=2><textarea name="message" rows="15" cols="55"></textarea></td></tr>
  <tr style="text-align: center" height=40><td colspan=2><input type="submit" value="Submit" /></td></tr>
  </table>
  </form>
<?php
}

function refresh_page($pg_now, $pg_to, $pg_url, $passing_var, $passing_arg){
	echo "<h3 style='text-align:center'>".$pg_now." successful.</h3>";
	echo "<p style='text-align:center'><span class='spec'> Now loading '".$pg_to."' page.. 
		  <br/></span></p>";
	echo '<meta http-equiv="refresh" content= "2;
				   URL= \''.$pg_url.'?'.$passing_var.'='.$passing_arg.'\'">';
}

function do_header($display_user_menu = true) {
  // print an HTML header
	@session_start();
?>
  <html>
  <head>
    <title>StatQueue</title>
	<link rel="stylesheet" type="text/css" href="/stylesheets/styles.css" />
  </head>
  <body>
	<!-- page header -->
	<table bgcolor = #0b215b width = 100% cellpadding="4" cellspacing="4">
	  <tr><td></td></tr>
	  <tr><td align = center colspan = 4><t1 style="text-align: center">StatQueue</t1></td></tr>
	  <tr><td></td></tr>
	</table>
	<!-- menu -->
	<br/>
	<table align = center bgcolor = white width = 45% cellpadding="4" cellspacing="4">
	  <tr align="center">
		<td width="10%"><a href="/pages/home.php"><img src="/images/home.png" border="0" width="18" height="18"> Home</a></td>
		<td width="10%"><a href="/pages/about.php"><img src="/images/about.png" border="0" width="18" height="18"> About</a></td>
		<td width="10%"><a href="/forum/"><img src="/images/forum.png" border="0" width="18" height="18"> Forum</a></td>
		<td width="10%"><a href="/db_entry/search_form.php"><img src="/images/search.png" border="0" width="18" height="18"> Search</a></td>
	  </tr>  
	</table>
	<br />
	<hr />
<?php
	#code below displays user menu.
	if ($display_user_menu == true){
		if (isset($_GET['expand_user_menu'])){
			$_SESSION['expand_user_menu'] = $_GET['expand_user_menu'];
		}
		if (isset($_SESSION['expand_user_menu'])){
			display_user_menu($_SESSION['expand_user_menu']);
		} else {
			display_user_menu();
		}
	}
}

function display_user_menu($expand = false){

    $user = check_valid_user();
    $db = db_connect();
    
	if ($user){
		#master has all options, admin has admin and user options, user has user option only.
		$userid = $_SESSION[$user];
		$_SESSION['userid'] = $userid;
		
		$welcome_stm;
		$user_option = "<a href='/db_entry/user_profile.php'>My Profile</a><br />
					   	<a href='/user_auth/change_pw.php'>Change password</a><br />";
		if(patient::check_if_on_wl($db, $userid)){
			$user_option.="<a href='/clinic/my_wl_status.php'>My waitlist status</a><br />";
		}
		$master_option = "";
		$admin_option = "";
		$logout_option = "<a href='/user_auth/logout.php'>Log out</a></h5>";
		
		if ($user == 'master'){
			$user = $_SESSION['master'];
			$master_option = "<a href='/db_entry/clinic_form.php'>Add new clinic</a><br />";
			$admin_option = "<a href='/clinic/current_wl.php'>View waitlist</a><br />";
		} else if ($user == 'admin') {
			$user = $_SESSION['admin'];
			$admin_option = "<a href='/clinic/current_wl.php'>View waitlist</a><br />
		  					 <a href='/db_entry/doctors_profile.php'>View doctors</a><br />";
		} else {
			$user = $_SESSION['user'];
		}
		
		$query = "select p.patientname from authorized_users as u, patients as p
					  where u.patientid = p.patientid and u.userid = '".$user."'";
		$result = mysqli_query($db, $query);
		$row = mysqli_fetch_assoc($result);
		
		$name_array = explode(" ", $row['patientname']);
		$firstname = $name_array[0];
		
		$welcome_stm = "<strong> Welcome, ".$firstname."</strong><br/>";
		if ($expand==true){
			$menu_icon = "<h5 style='text-align:right'><a href='?expand_user_menu=0'>
					 	  <img src='/images/menu.png' width=18 height=18></a>";
			$user_menu = $menu_icon.$welcome_stm.$user_option.
						 $master_option.$admin_option.$logout_option."</h5>";
		} else {
			$menu_icon = "<h5 style='text-align:right'><a href='?expand_user_menu=1'>
					 	  <img src='/images/menu.png' width=18 height=18></a>";
			$user_menu = $menu_icon.$welcome_stm."</h5>";
		}
		echo $user_menu;
	}
}

function do_footer() {
?>
	<!-- page footer -->
	<br /><br /><br />
	<hr />
	<div class="footer">
	Copyright &copy; 2015 StatQueue(www.statqueue.com). All rights
	reserved. Used with permission.<br /> 
	<a href="mailto:hongman.cho@gmail.com?subject=Question&
			 body=When will the website be available for use?">
			 support@statqueue.com</a>
	</div>
	</body>
	</html>
<?php
}

function display_login_form() {
?>
  <h2>Log In</h2>
  <p>Enter your Touch In user ID and password below.</p>
  <form method="post" action="login.php">
  <table bgcolor="#cccccc" cellspacing=2 cellpadding=2>
  <tr><td>Userid:</td>
  <td><input type="text" name="userid"></td></tr>
  <tr><td>Password:</td>
  <td><input type="password" name="password"></td></tr>
  <tr><td colspan="2" align="center">
  <input type="submit" value="Log In"></td></tr>
  <tr><td colspan="2">
  <span class=spec><a href="forgot_pw.php">Forgot your password?</a></span></td></tr>
  <tr><td colspan="2">
  <span class=spec><a href="/db_entry/register_form.php">Not a user?</a></span></td></tr>
  </table>
  </form>
<?php
}

function display_registration_form() {
?>
	<p><br/><strong>Please type in your account info below and submit when finished to create your account.</strong></p>
	<span class=spec>*Upon account creation, a message including your account info and patient ID
	will be sent to your email.<br/>The patient ID is used for recalling your patient info
	to conviniently sign-up for waitlist in future.<br/><br/></span>
	<table width=700 cellpadding=5 cellspacing=0 bgcolor=#cccccc>
	<form method="post" action="register_new.php">
	<tr><td>
		<div class="formlabel">
			Accountname:<br />
		</div>
		<div class="formfield">
			<input type="text" name="accountname" size="30" />
		</div>
		<div class="formlabel">
			Password:<br /> <span class="spec">*Your password
				length must be 6 at least and it must contain at least one lower
				case letter, upper case letter, and number.</span>
		</div>
		<div class="formfield">
			<input type="password" name="password" size="30" />
		</div>
		<div class="formlabel">Verify Password:</div>
		<div class="formfield">
			<input type="password" name="password2" size="30" />
		</div>
		<div class="formlabel">Firstname:</div>
		<div class="formfield">
			<input type="text" name="firstname" size="25" />
		</div>
		<div class="formlabel">Lastname:</div>
		<div class="formfield">
			<input type="text" name="lastname" size="25" />
		</div>
		<div class="formlabel">
			Date of Birth:<br /> <span class="spec">*Enter
				in a form of '1979/07/21'.</span>
		</div>
		<div class="formfield">
			<input type="text" name="birthdate" size="25" />
		</div>
		<div class="formlabel">Gender:</div>
		<div class="formfield">
			<input type="radio" name="gender" value="male" />Male<br />
			<input type="radio" name="gender" value="female" />Female
		</div>
		<div class="formlabel">E-mail Address:</div>
		<div class="formfield">
			<input type="text" name="email" size="30" />
		</div>
		<div class="formlabel">Phone Number:</div>
		<div class="formfield">
			<input type="text" name="phonenumber" size="20" />
		</div>
		<div class="formlabel">Street Address:</div>
		<div class="formfield">
			<input type="text" name="streetaddress" size="60" />
		</div>
		<div class="formlabel">City:</div>
		<div class="formfield">
			<input type="text" name="city" size="30" />
		</div>
		<div class="formlabel">Province:</div>
		<div class="formfield">
			<select name="province">
      		<option value="Ontario">Ontario</option>
      		<option value="Quebec">Quebec</option>
    		</select>
		</div>
		<div class="formlabel">Postal Code:<br /> <span class="spec">*without a space.</span>
		</div>
		<div class="formfield">
			<input type="text" name="postalcode" size="10" />
		</div>
		<div class="formlabel">Health Card Number:<br /> <span class="spec">*Without space or hyphen.</span>
		</div>
		<div class="formfield">
			<input type="text" name="healthcardnumber" size="30" />
		</div>
		<div class="formlabel">Medical History:<br /> <span class="spec">
		*Provide your medical history that doctors should know beforehand for your
		 treatment.</span>
		</div>
		<div class="formfield">
			<input type="text" name="medicalhistory" size="60" />
		</div>
		<div class="formlabel">Clinic ID:<br /> <span class="spec">
		*If you are a clinic administrator, please enter the Clinic ID of clinic at which you work.</span>
		</div>
		<div class="formfield">
			<input type="text" name="clinicid" size="10" />
		</div>
		</td></tr>
		<tr bgcolor=white><td><br/>
		<div style="text-align: center; clear: left">
			<input type="submit" value="click Here to Submit" />
			<input type="reset" value="Erase and Start Over" />
		</div>
		</td></tr>
	</form>
	</table>
	<br />
<?php
}

function display_pw_form() {
  // display html change password form
?>
   <h2>Change Password</h2>
   <br />
   <form action="change_pw.php" method="post">
   <table width="350" cellpadding="5" cellspacing="0" bgcolor="#cccccc">
   <tr><td width=45% align = right><strong>Old Password:</strong></td>
       <td><input type="password" name="old_pw"
            size="20" maxlength="20"/></td>
   </tr>
   <tr><td align = right><strong>New Password:</strong></td>
       <td><input type="password" name="new_pw"
            size="20" maxlength="20"/></td>
   </tr>
   <tr><td align = right><strong>Repeat New Password:</strong></td>
       <td><input type="password" name="new_pw2"
            size="20" maxlength="20"/></td>
   </tr>
   <tr height=40><td colspan="2" align="center" bgcolor = white>
       <input type="submit" value="Submit"/>
   </td></tr>
   </table>
   </form>
   <br />
<?php
}

function display_forgot_form() {
  // display HTML form to reset and email password
?>
   <h3>Lost Password Form</h3>
   <span class=spec>*Enter your accountname and associated email address below.<br/><br/></span>
   <form action="forgot_pw.php" method="post">
   <table width="350" cellpadding="5" cellspacing="0" bgcolor="#cccccc">
   <tr><td>User ID:</td>
       <td><input type="text" name="userid" size="16" maxlength="16"/></td>
   </tr>
   <tr><td>Email:</td>
       <td><input type="text" name="email" size="30" maxlength="50"/></td>
   </tr>
   <tr><td colspan=2 align="center">
       <input type="submit" value="Change password"/>
   </td></tr>
   </table>
   </form>
   <br />
<?php
}

class profile{
	
	public $db, $userid, $row, $display_url, $update_url;
	
	function __construct($userid){
		$this->db = db_connect();
		$query = "select adminid, patientid from authorized_users where userid = '".$userid."'";
		$result = mysqli_query($this->db, $query);
		$row = mysqli_fetch_assoc($result);
		
		$query = "select * from patients where patientid = ".$row['patientid'];
		$result = mysqli_query($this->db, $query);
		$this->row = mysqli_fetch_assoc($result);
		
		if ($row['adminid'] != 0){
			$query= "select c.clinicname, a.clinicid from clinicadmins as a, clinics as c
					 where a.adminid = ".$row['adminid']."
  					 and a.clinicid = c.clinicid";
			$result = mysqli_query($this->db, $query);
			$clinic_row = mysqli_fetch_assoc($result);
			$this->row['adminid'] = $row['adminid'];
			$this->row['clinicid'] = $clinic_row['clinicid'];
			$this->row['clinicname'] = $clinic_row['clinicname'];
		}
		$this->display_url = '/db_entry/user_profile_form.php';
		$this->edit_url = '/db_entry/user_profile_edit.php';
	}
	
	function display_profile(){
	?>
		<br />
		<h2>Profile info</h2>
		<table width="500" cellpadding="5" cellspacing="0" bgcolor="#cccccc">
	<?php
		foreach ($this->row as $key => $value){
			echo '<tr><td style = "text-align: right" width=10%><strong>'.$key.': </strong></td>
					  <td>'.$value.'</td>';
		}
		if (isset($this->row['patientid'])){
			$id = $this->row['patientid'];
		} else if (isset($this->row['doctorid'])){
			$id = $this->row['doctorid'];
		}
		echo '<tr bgcolor=white height=60><td style="text-align:center" colspan=3>
	  		  <a href = "'.$this->display_url.'"><img src="/images/edit2.png" 
          	   border="0" width="70" height="35"></a></td></tr></table><br />';
	}
	
	function display_profile_edit_form(){
		echo '<br />
		<h2>Edit Profile</h2>';
		if (isset($this->doctorid)){
			echo '<span class=spec>*Editing doctor\'s schedule would reset 
					availability based on edited schedule and current Eastern Time.<br/><br/></span>';
		}
		echo '<form action='.$this->edit_url.' method="post">
				<table width="500" cellpadding="5" cellspacing="0" bgcolor="#cccccc">';

		foreach ($this->row as $key => $value){
			echo '<tr><td style="text-align:right" width=10%><strong>'.$key.': </strong>';
			if ($key == 'schedule'){
				echo '<br/><a href = "'.$this->display_url.'?schedule=true">
				  	  <img src="/images/edit2.png" border=0 width=40 height=21 alt="edit schedule"></a>
					  </td>
					  <td>'.$value.'</td>
					  </tr>';
			} else if ($key == 'patientid' || $key == 'patientname' || $key == 'healthcardnumber' ||
					   $key == 'adminid' || $key == 'doctorid' || $key == 'doctorname' ||
					   $key == 'birthdate' || $key == 'gender' || $key == 'clinicname' ||
					   $key == 'clinicid' || $key == 'patients'){
				echo '</td>
					  <td>'.$value.'</td>
					  </tr>';
			} else {
				echo '</td>
					  <td><input type="text" name="'.$key.'" size="40" value="'.$value.'"/></td>
					  </tr>';
			}
		}
		echo '<tr bgcolor = white><td colspan=2 align="center" height=42>';
		if (isset($this->doctorid)){
			echo '<input type=hidden name=doctorid value='.$this->doctorid.'>';
		}
		echo '
		<input type="submit" value="submit"/></td></tr>
		</table>
		</form>
		<br />';
	}
}

class doctors_profile extends profile{
	public $doctor_info, $doctorid;
	
	function __construct($doctorid){
		$this->doctorid = $doctorid;
		$this->db = db_connect();
		$query = "select * from doctors where doctorid = ".$doctorid;
		$result = mysqli_query($this->db, $query);
		$this->row = mysqli_fetch_assoc($result);
		$this->row['schedule'] = nl2br($this->row['schedule']);
		$this->display_url = '/db_entry/doctors_profile_form.php';
		$this->edit_url = '/db_entry/doctors_profile_edit.php';
	}
}

class patient_profile extends profile{
	public $patient_info, $patientid;

	function __construct($patientid){
		$this->patientid = $patientid;
		$this->db = db_connect();
		$query = "select * from patients where patientid = ".$patientid;
		$result = mysqli_query($this->db, $query);
		$this->row = mysqli_fetch_assoc($result);
		$this->display_url = '/db_entry/patient_profile_form.php';
		$this->edit_url = '/db_entry/patient_profile_edit.php';
	}
}

function display_search_form($postal_code){
?>
  <h2 style="text-align: center">StatQueue Clinic Search</h2><br />
  <form action="search_result.php" method="post">
    <p style="text-align: center"><strong>Choose Search Type:</strong><br /><br />
    <select name="searchtype">
      <option value="clinic">Clinic</option>
      <option value="treatment">Treatment</option>
      <option value="doctor">Doctor</option>
    </select>
    <br /><br /><strong>Enter Search Term:</strong><br /><br />
    <input name="searchterm" type="text" size="40"/>
    <br /><br />
    <input type="submit" name="submit" value="Search"/>
    <br /><br /><strong>
    or
    </strong><br /><br />
<?php
	echo "<a href = '/db_entry/search_result.php?postalcode=".$postal_code."'>Show all in my town</a>";
?>
    </p>
  </form>
<?php
}


function display_search_results($db, $searchtype, $searchterm, $operator = 'like'){
	#this query does not ditinguish same clinicid if there are more than one doctors in the clinic.
	$query = "select distinct c.clinicname, c.clinicid, c.treatment, c.waitlistlength
		from clinics as c, doctors as d
		where ".$searchtype." ".$operator." '".$searchterm."'
		and c.clinicid = d.clinicid
		order by c.clinicid asc";

	$result = mysqli_query($db, $query);
	$num_results = mysqli_num_rows($result);

	echo "<h2>Search Results</h2>
		  <p><strong>Number of matching results found: ".$num_results."</strong></p><br/>";
	
	$clinics = array();
	for ($i = 1; $i <= $num_results; $i++){
		$row = mysqli_fetch_assoc($result);
		foreach($row as $key=>$value){
			$value = htmlspecialchars(stripslashes($value));
/* 			if ($key == 'clinicname'){
				array_push($clinics, $value.'+clinic');
			} */
		}
		echo "<table width=350 bgcolor=#cccccc cellpadding=10 cellspacing=0><tr><td>
  			  <p><strong>".($i).". Clinic: ".$row['clinicname']."</strong><br/>
			  <br />Treatments: ".$row['treatment']."
			  <br />Current number of waitlisted patients: ".$row['waitlistlength']."</p>
			  </td></tr>
			  <tr align=center bgcolor=white height=60><td>
			  <a href = '/db_entry/doctors_profile.php?clinicid=".$row['clinicid']."'>
			  <img src='/images/doctorlist.png' width=102 height=30></a>
			  <a href = '/clinic/wl_signup_form.php?clinicid=".$row['clinicid']."'>
			  <img src='/images/signup.png' width=85 height=30></a>
			  </td></tr></table><br/><br/>";
	}
/* 	$clinics = implode(',', $clinics);
	echo '<iframe width="600" height="450" frameborder="0" style="border:0"
  	src="https://www.google.com/maps/embed/v1/search?key=AIzaSyDWOYkC4iLGVJqGxApuzetmYnQ7127cpcg
  	&q='.$clinics.'+in+toronto+ON">
	</iframe>'; */
}


function display_patient_form(){
?>
<h2>Patient Entry for Waitlist</h2>
<br />
<form method="post" action="patient_new.php">
	<div class="formlabel">Firstname:</div>
	<div class="formfield">
		<input type="text" name="firstname" size="25" />
	</div>
	<div class="formlabel">Lastmame:</div>
	<div class="formfield">
		<input type="text" name="lastname" size="25" />
	</div>
	<div class="formlabel">
	Date of Birth:<br /> <span class="spec">
	*Enter in a form of '1979/07/21'.</span>
	</div>
	<div class="formfield">
		<input type="text" name="birthdate" size="15" />
	</div>
	<div class="formlabel">Gender:</div>
	<div class="formfield">
		<input type="radio" name="gender" value="male" />Male<br />
		<input type="radio" name="gender" value="female" />Female
	</div>
	<div class="formlabel">E-mail Address:</div>
	<div class="formfield">
		<input type="text" name="email" size="30" />
	</div>
	<div class="formlabel">Phone Number:</div>
	<div class="formfield">
		<input type="text" name="phonenumber" size="20" />
	</div>
	<div class="formlabel">Street Address:</div>
	<div class="formfield">
		<input type="text" name="streetaddress" size="100" />
	</div>
	<div class="formlabel">City:</div>
	<div class="formfield">
		<input type="text" name="city" size="25" />
	</div>
	<div class="formlabel">Province:</div>
	<div class="formfield">
		<select name="province">
    	<option value="Ontario">Ontario</option>
    	<option value="Quebec">Quebec</option>
    	</select>
	</div>
	<div class="formlabel">Postal Code:<br />
	<span class="spec">*Without a space.</span>
	</div>
	<div class="formfield">
		<input type="text" name="postalcode" size="10" />
	</div>
	<div class="formlabel">
	Health Card Number:<br /> <span class="spec">
	*Without space or hyphen.</span>
	</div>
	<div class="formfield">
		<input type="text" name="healthcardnumber" size="20" />
	</div>
	<div class="formlabel">Medical History:<span class="spec"><br />
	*Type 'none' if not specified.</span>
	</div>
	<div class="formfield">
		<input type="text" name="medicalhistory" size="100" />
	</div>
	<div class="formlabel">Treatment:<span class="spec"><br />
	*Type of exam/treatment you seek.</span></div>
	<div class="formfield">
		<input type="text" name="treatment" size="30" />
	</div>
	<div class="formlabel">Doctor:<span class="spec"><br />
	*Fullname of the doctor for whom you wish to sign up.
	(enter either first or last if fullname could not be found)
	Leave blank otherwise.</span></div>
	<div class="formfield">
		<input type="text" name="doctor" size="25" />
	</div>
	<div style="text-align: left; clear: left">
		<input type="submit" value="click Here to Submit" /> <input
			type="reset" value="Erase and Start Over" />
	</div>
</form>
<br />
<?php
}

function display_simple_wl_form(){
	?>
	<h2>Simple Patient Entry for Waitlist</h2>
	<br />
	<form method="post" action="simple_patient_new.php">
		<div class="formlabel">Patient ID: </div>
		<div class="formfield">
			<input type="text" name="patientid" size="25" />
		</div>
		<div style="text-align: left; clear: left">
		*Enter indented values below if patient ID is not available.<br/><br/>
		</div>
		<div class="formlabel" style="margin-left:20">Firstname:</div>
		<div class="formfield">
			<input type="text" name="firstname" size="25" />
		</div>
		<div class="formlabel" style="margin-left:20">Lastname:</div>
		<div class="formfield">
			<input type="text" name="lastname" size="25" />
		</div>
		<div class="formlabel" style="margin-left:20">
		Date of Birth:<br /> <span class="spec">
		*Enter in a form of '1979/07/21'.</span>
		</div>
		<div class="formfield">
			<input type="text" name="birthdate" size="15" />
		</div>
		<div class="formlabel" style="margin-left:20">Gender:</div>
		<div class="formfield">
			<input type="radio" name="gender" value="male" />Male<br />
			<input type="radio" name="gender" value="female" />Female
		</div>
		<div class="formlabel" style="margin-left:20">Medical History:
		<span class="spec"><br />*Type 'none' if not specified.</span>
		</div>
		<div class="formfield">
			<input type="text" name="medicalhistory" size="75" />
		</div>
		<div class="formlabel">Treatment:<span class="spec"><br />
		*Type of exam/treatment you seek.</span></div>
		<div class="formfield">
			<input type="text" name="treatment" size="30" />
		</div>
		<div class="formlabel">Doctor:<span class="spec"><br />
		*Fullname of the doctor for whom you wish to sign up.
		(enter either first or last if fullname could not be found)
		Leave blank otherwise.</span></div>
		<div class="formfield">
			<input type="text" name="doctor" size="25" />
		</div>
		<div style="text-align: left; clear: left; margin-left: 150">
			<input type="submit" value="Submit" /> <input
				type="reset" value="Erase" />
		</div>
	<br />
	</form>
	<?php
}

function display_wl_signup_form($clinicid, $doctorid = false){
	?>
	<form method="post" action="signup_wl.php">
	<br />
	<table width=580 bgcolor=#cccccc cellspacing=0 cellpadding=5>
	<tr>
	<td width=40% style="text-align:right"><strong>Treatment: </strong><br /> <span class="spec">
	*Briefly describe the type of your illness or the type of examination you seek.</span></td>
	<td><input type="text" name="treatment" size="50" />
	<input type=hidden name=clinicid value=<?php echo $clinicid;?>>
	<?php
	if ($doctorid != false){
	echo '<input type=hidden name=doctorid value='.$doctorid.'></td></tr>';
	} else {
	echo '</td></tr>
  		<tr>
		<td width=40% style="text-align:right"><strong>Doctor: <br /> </strong><span class="spec">
		*Enter the fullname of doctor if specified. Leave blank if otherwise.</span></td>
		<td><input type="text" name="doctor" size="25" />
		<input type=hidden name=clinicid value='.$clinicid.'></td>
		</tr>';
	}
	?>
	<tr height=40 bgcolor=white>
	<td style="text-align:center" colspan=2>
	<input type="submit" value="Submit" /> <input type="reset" value="Erase" /></td>
	</tr>
	</table>
	</form>
	<br />
	<?php
}

function display_doctors_appointment_form(){
	?>
	<br />
	<form method="post" action="signup_wl.php">
	<table width=570 bgcolor=#cccccc cellspacing=0 cellpadding=5>
	<tr>
	<td width=40% style="text-align:right"><strong>Treatment: </strong><br /> <span class="spec">
	*Briefly describe the type of your illness or the type of examination you seek.</span></td>
	<td><input type="text" name="treatment" size="30" /></td>
	</tr>
	<tr height=40 bgcolor=white>
	<td style="text-align:center" colspan=2><input type="submit" value="Submit" /> <input type="reset" value="Erase" /></td>
	</tr>
	</table>
	</form>
	<br />
	<?php
}

function display_my_wl_status($db, $patientid){
	$query = "select wl.waitlistid, wl.clinicid, wl.patientname, 
			  wl.treatment, wl.doctorname, wl.order, c.waitlistlength
			  from waitlist as wl left join clinics as c on wl.clinicid = c.clinicid
			  where wl.patientid = ".$patientid;
	$result = mysqli_query($db, $query);
	$num_rows = mysqli_num_rows($result);

	echo "<h2>My current waitlist status</h2>
		  <span class='spec'>*Users can sign-up <strong>upto two waitlists</strong>
  		   concurrently at most.<br/>
		   This page automatically refreshes itself every 1 minute.<br/>
  		   Delist option is available until <strong>user's queue is 5</strong> or
  		   estimated waiting time has yet passed below <strong>30 minutes</strong>.
		  </span>";
	for($i = 0; $i < $num_rows; $i++){
		$row = mysqli_fetch_assoc($result);
		$clinicname = clinic::get_clinicname($db, $row['clinicid']);
		echo '
		<br />
		<p><strong>'.($i+1).'. '.$clinicname.'</strong><p>
		<table width="1100" cellpadding="5" cellspacing="0" bgcolor="#cccccc">
		<tr>
		<td width = "15%" style = "text-align: center"><strong>Name</strong></td>
		<td width = "25%" style = "text-align: center"><strong>treatment</strong></td>
		<td width = "15%" style = "text-align: center"><strong>doctor</strong></td>
		<td width = "8%" style = "text-align: center"><strong>my queue</strong></td>
		<td width = "10%" style = "text-align: center"><strong>wait time</strong></td>
		<td width = "10%" style = "text-align: center"><strong>list length</strong></td>
		<td style = "text-align: center"><strong>delist</strong></td>
		</tr>
		
		<tr>
		<td style = "text-align: center">'.$row['patientname'].'</td>
		<td style = "text-align: center">'.$row['treatment'].'</td>
		<td style = "text-align: center">'.$row['doctorname'].'</td>
		<td style = "text-align: center">'.$row['order'].'</td>
		<td style = "text-align: center"></td>
		<td style = "text-align: center">'.$row['waitlistlength'].'</td>';
		if ($row['order'] >= 5){
			echo '
			<td style = "text-align: center" width = 5%>
			<a href="/clinic/delist_wl.php?clinicid='.$row['clinicid'].'&
			wl_id='.$row['waitlistid'].'&wl_length='.$row['waitlistlength'].'" 
			onclick="return confirm(\'Are you sure you wish to delist yourself form this waitlist?\')">
			<img src="/images/delete.png" width=15 height=15 alt="delist"></a></td>';
		}
    	echo '
		</tr>
    	</table>
		<br />';
	}
}

function display_doctor_form(){
	?>
	<h2>Add New Doctor to your clinic</h2>
	<br />
	<form method="post" action="doctor_new.php">
		<div class="formlabel">Firstname:</div>
		<div class="formfield">
			<input type="text" name="firstname" size="25" />
		</div>
		<div class="formlabel">Lastname:</div>
		<div class="formfield">
			<input type="text" name="lastname" size="25" />
		</div>
		<div class="formlabel">
		Date of Birth:<br /> <span class="spec">
		*Enter in a form of '1979/07/21'.</span>
		</div>
		<div class="formfield">
			<input type="text" name="birthdate" size="15" />
		</div>
		<div class="formlabel">Gender:</div>
		<div class="formfield">
			<input type="radio" name="gender" value="male" />Male<br />
			<input type="radio" name="gender" value="female" />Female
		</div>
		<div class="formlabel">E-mail Address:</div>
		<div class="formfield">
			<input type="text" name="email" size="30" />
		</div>
		<div class="formlabel">Phone Number:</div>
		<div class="formfield">
			<input type="text" name="phonenumber" size="20" />
		</div>
		<div class="formlabel">Specialty</div>
		<div class="formfield">
			<input type="text" name="specialty" size="30" />
		</div>
		<div style="text-align: left; clear: left">
			<input type="submit" value="click Here to Submit" /> <input
				type="reset" value="Erase and Start Over" />
		</div>
	</form>
	<br />
	<?php
}

function display_doctors_schedule_form($db, $doctorid){
	$query = "select schedule from doctors where doctorid = ".$doctorid;
	$result = $db->query($query);
	$row = $result->fetch_row();
?>
	<h2>Schedule form</h2>
	<span class='spec'>Please enter hours in number. If you are entering an afternoon(p.m.) number, add 12 to it.<br />
	   Ex. enter 12 for 12pm; enter 16 for 4pm; enter 9 for 9am, etc.<br/>
	   *Enter 0 in both boxes on days in which the doctor is not available.<br/><br/></span>
	<form method="post" action="doctors_schedule_new.php">
	<table width="300" cellpadding="5" cellspacing="0" bgcolor="#cccccc">
<?php 
	$schedule = preg_split("/[\n ]+/", $row[0]);
	
	for($i=0; $i < count($schedule)-1; $i+=5){
		echo '<tr><td align = "right">'.$schedule[$i].'</td><td> from 
		<input style="text-align:right" type="int" maxlength="2" size="1" name='.$i.'from value ='.$schedule[$i+2].'> to 
		<input style="text-align:right" type="int" maxlength="2" size="1" name='.$i.'to value ='.$schedule[$i+4].'>
		</td></tr>';
	}
	echo '
	<tr><td colspan = "2" align = "center">
  	<input type=hidden name=doctorid value='.$doctorid.'>
  	<input style = "text-align:right" type = "submit" value = "submit"></td></tr>
	</table>
	</form>
	<br />';
}

function display_clinic_form(){
	$info_list = array('Clinic Name:'=>'clinicname', 'Street Adress:'=>'streetaddress', 
					   'City:'=>'city', 'Postal Code:'=>'postalcode', 'Email Address:'=>'email',
					   'Phone Number:'=>'phonenumber', 'Treatment(s):'=>'treatment');
	echo '<h2>New Clinic Form</h2>
		  <form action="clinic_new.php" method="post">
		  <table cellpadding="5" cellspacing="0" bgcolor=#cccccc width=560>';
	foreach($info_list as $key => $value){
		echo '<tr>
	        <td width=30% style="text-align:right"><strong>'.$key.'</strong></td>
	        <td><input type="text" name='.$value.' maxlength="100" size="50"></td>
	      </tr>';
		if ($key == 'City:'){
			echo '<tr><td width=30% style="text-align:right"><strong>Province</strong></td>
				   	<td>
					<select name="province">
					<option value="Ontario">Ontario</option>
					<option value="Quebec">Quebec</option>
					</select>
					</td></tr>';
		}
	}
    echo '<tr bgcolor=white>
		  <td colspan="2" height=40 style="text-align: center"><input type="submit" value="Register"></td>
		  </tr>
		  </table>
		  </form>';
}

class clinic {
	
	public $clinicid, $db, $num_rows, $rows, $doctors_in, $wl_length;
	
	function __construct($db, $clinicid){
		$this->clinicid = $clinicid;
		$this->db = $db;
		$this->rows = array();
		$this->doctors_in = array();
		$this->query_wl();
		$this->wl_length = $this->get_wl_length($clinicid);
	}
	
	static function register_clinic($db, $clinic_info){
		$query = "insert into clinics values (null, '";
		$query .= implode("', '", $clinic_info)."', 0)";
		$result = mysqli_query($db, $query);
	
		if ($result){
			echo "Clinic, '".$clinic_info['clinicname']."' is inserted into database.";
		} else {
			throw new Exception ("Clinic could not be registered.");
		}
	}
	
	static function check_clinic($db, $clinicname){
		$query = "select * from clinics where clinicname = '".$clinicname."'";
		$result = mysqli_query($db, $query);
		$num_rows = mysqli_num_rows($result);
		if ($num_rows > 0){
			return true;
		} else {
			return false;
		}
	}
	
	static function get_clinicname($db, $clinicid){
		$query = "select clinicname from clinics where clinicid = ".$clinicid;
		$result = mysqli_query($db, $query);
		$row = mysqli_fetch_assoc($result);
		return $row['clinicname'];
	}
	
	function lookup_patient($patientid){
		$query = "select * from waitlist where
				  clinicid = ".$this->clinicid."
    			  and patientid = ".$patientid;
		$result = mysqli_query($this->db, $query);
		$num_rows = mysqli_num_rows($result);
		if ($num_rows > 0){
			$row = mysqli_fetch_assoc($result);
			return $row;
		} else {
			return false;
		}
	}
	
	function query_wl(){
		#rows order is reset evertime wl is queried.
		$query = "select waitlistid, patientid, patientname, birthdate,
				gender, medicalhistory, treatment, doctorname from waitlist
				where clinicid = ".$this->clinicid." order by waitlist.order asc";
		$result = mysqli_query($this->db, $query);
		$this->num_rows = mysqli_num_rows($result);
		$this->rows = array();
		for ($i = 1; $i <= $this->num_rows; $i++){
			$row = mysqli_fetch_row($result);
			array_push($this->rows, $row);
		}
	}
	
	function list_patient($info_list){
		#order is set based on length of rows
		if (isset($info_list['patientid'])){
			$patientid = array_pop($info_list);
			$info_list['order'] = count($this->rows)+1;
			$cols = "(clinicid, patientid, patientname, birthdate, gender, 
					  medicalhistory, treatment, doctorname, waitlist.order)";
			$query = "insert into waitlist ".$cols." values (".$this->clinicid.", ".$patientid.", '";
		} else {
			$info_list['order'] = count($this->rows)+1;
			$cols = "(clinicid, patientname, birthdate, gender, 
					  medicalhistory, treatment, doctorname, waitlist.order)";
			$query = "insert into waitlist ".$cols." values (".$this->clinicid.", '";
		}
		$string = implode("', '", $info_list);
		$query .= $string."')";
		$result = mysqli_query($this->db, $query);
		if (!$result){
			throw new Exception("patient could not be added to clinic's waitlist.
								 Please try again later.");
		}
		$this->query_wl();
		$this->wl_length++;
		$this->update_wl_length($this->clinicid, $this->wl_length);
		doctor::update_num_patients($this->db, $info_list['doctor'], +1);

		return true;
	}
	
	static function get_wl_length($clinicid){
		$db = db_connect();
		$query = "select waitlistlength from clinics where clinicid=$clinicid";
		$result= $db->query($query);
		return $result->fetch_row()[0];
	}
	
	static function update_wl_length($clinicid, $wl_length){
		$db = db_connect();
		$query = "update clinics set waitlistlength=$wl_length
				  where clinicid=$clinicid";
		$result = $db->query($query);
		if(!$result){
			throw new Exception("waitlist length could not be updated.");
		}
		return true;
	}
	
	static function unlist_patient($db, $clinicid, $wl_id, $wl_length){
		$doctor = doctor::get_doctorname($db, $wl_id);
		doctor::update_num_patients($db, $doctor, -1);
		
		clinic::update_wl_length($clinicid, $wl_length-1);
		$query = "select waitlist.order from waitlist
				  where clinicid=$clinicid
				  and waitlist.order > 
				  (select waitlist.order from waitlist
				   where clinicid=$clinicid
				   and waitlistid=$wl_id)";
		$result = $db->query($query);
		
		for ($i=0; $i < @$result->num_rows; $i++){
			$order = $result->fetch_row()[0];
			$query = "update waitlist set waitlist.order=".($order-1)."
  					  where waitlist.order = $order
					  and clinicid=$clinicid";
			$result = $db->query($query);
		}
		if (!$result){
			throw new Exception("waitlist order could not be reset.");
		}
		
		$query = "delete from waitlist where waitlistid = ".$wl_id;
		$result = mysqli_query($db, $query);
		if (!$result){
			throw new Exception("patient could not be unlisted from your clinic's waitlist.
								 Please try again later.");
		}
		
		return true;
	}
	
	function display_current_wl(){
		#must query wl at start since a patient could have listed/delisted herself.
		$this->query_wl();
	?>
		<br />
		<h2>Current waitlist of <?php echo clinic::get_clinicname($this->db, $this->clinicid);?></h2>
		<span class="spec">*This page automatically updates itself every minute.<br/><br/></span></div>
		<form action = "/clinic/update_wl.php" method = "post">
		<table align = "center" width="95%" cellpadding="5" cellspacing="0" bgcolor="#cccccc">
			<tr style = "text-align: center">
				<td><strong>order</strong></td>
				<td><strong>patient ID</strong></td>
				<td><strong>patient name</strong></td>
				<td><strong>birth date</strong></td>
				<td><strong>gender</strong></td>
				<td><strong>medical history</strong></td>
				<td><strong>treatment</strong></td>
				<td><strong>doctor name</strong></td>
				<td><strong>delete</strong></td>
				<td><strong>Profile</strong></td>
			</tr>
	<?php
		for ($i = 0; $i < count($this->rows); $i++){
			echo "<tr><td style = 'text-align: center'>".($i+1)."</td>";
			for ($j = 1; $j < count($this->rows[$i]); $j++){
				echo "<td style = 'text-align: center'>".$this->rows[$i][$j]."</td>";
			}
			$wl_id = $this->rows[$i][0];
			$patientid = $this->rows[$i][1];
			
			echo "<td align='center'><input type='checkbox' name='del_me[]' value='".$wl_id."'></td>";
			if ($patientid != 0){
				echo "<td align = 'center'><a href = '/db_entry/patient_profile.php?patientid=".$patientid."'>
	  				  <img src='/images/view.png' width=45 height=20></a></td></tr>";
			}
		}
	?>
		<tr><td><br/></td></tr>
		<tr height=50 align=center><td colspan = 10>
		<input type = "hidden" name=clinicid value=<?php echo $this->clinicid;?>>
		<input type = "hidden" name=wl_length value=<?php echo $this->wl_length;?>>
		<input type = "submit" value = "delete"></td></tr>
		</table>
		</form>
		<br />
		
		<form action = "/clinic/update_wl.php" method = "post">
		<table align=center width="95%" cellpadding="5" cellspacing="0" bgcolor="white">
		<tr><td align = "center">change order from
		<input style = text-align:right type = text name = "order1" size = 1 max-length = 3> to
		<input style = text-align:right type = text name = "order2" size = 1 max-length = 3>
		</td></tr>
		<tr height=50><td align = "center">
		<input type = hidden name = clinicid value = <?php echo $this->clinicid;?>>
		<input type = "submit" value = "submit"></td></tr>
		<tr><td><br/></td></tr>
		<tr><td align = "center"><a href = "/db_entry/patient_form.php">list new patient</a></td></tr>
		<tr><td align = "center">or</td></tr>
		<tr><td align = "center"><a href = "/db_entry/simple_patient_form.php">quick sign up</a></td></tr>
		<tr><td><br/></td></tr>
		</table>
		</form>
		<br />
	<?php
	}
	
	function change_wl_order($rows, $order1, $order2){
		#move value from rows[i1] to rows[i2].
		$i1 = $order1 - 1;
		$i2 = $order2 - 1;
		$new_rows = array();
		foreach ($rows as $i=>$value){
			if ($i < $i1 && $i < $i2){
				array_push($new_rows, $rows[$i]);
			} else if ($i == $i2){
				array_push($new_rows, $rows[$i1]);
				array_push($new_rows, $rows[$i]);
			} else if ($i != $i1){
				array_push($new_rows, $rows[$i]);
			}
		}
		$this->rows = $new_rows;
		#now query the reordered wl orders for each listed patients;
		for ($i = 0; $i < count($this->rows); $i++){
			$patientid = $this->rows[$i][1];
			$query = "update waitlist set waitlist.order = ".($i+1)."
    				  where clinicid = ".$this->clinicid."
					  and patientid = ".$patientid;
			$result = $this->db->query($query);
			if (!$result){
				throw new Exception("waitlist order could not be updated.");
			}
		}
		$this->query_wl();
		return true;
	}
	
	function display_doctors($user){
	?>
		<br />
		<h2>Doctors at <?php echo clinic::get_clinicname($this->db, $this->clinicid);?></h2>
	<?php
		if ($user == 'user'){
			echo '<span class=spec>*When user makes an appointment with a doctor, user will be queued
			at the end of current waitlist of the clinic regardless<br/>of number of patients waiting
			for the doctor.<br/><br/>
			Ex. If current length of waitlist is 16 and user makes appointment with a doctor for whom
			4 patients are in line, user will<br/>be 17th in waitlist queue and is a 5th patient for
			the doctor. It is important to note that user is not subjected to be called<br/>by the 
			doctor until user is 1st in waitlist queue.<br/><br/></span>';
		} else {
			echo '
  			<span class=spec>*Click refresh button to reset availabilities of listed doctors  
  			 based on current time and their saved schedules.<br/><br/></span>';
		}
	?>
		<table width="650" cellpadding="5" cellspacing="0" bgcolor="#cccccc">
			<tr><td width = "50%" style = "text-align: center"><strong>full name</strong></td>
				<td width = "40%" style = "text-align: center"><strong>specialty</strong></td>
				<td style = "text-align: center"><strong>Availability</strong>
				<a href = "/db_entry/doctors_profile.php?refresh=true">
  				<img src="/images/refresh.png" width=58 height=20></a><br/></td>
	<?php				
				if($user!='user'){ 
					echo '<td style = "text-align: center"><strong>Profile</strong></td>
						  <td style = "text-align: center"><strong>Delete</strong></td></tr>';
				} else {
					echo '<td style = "text-align: center"><strong>Patients</strong></td>
  						  <td style = "text-align: center"><strong>Appointment</strong></td></tr>';
				}

		$query = "select * from doctors where clinicid = ".$this->clinicid;
		$result = mysqli_query($this->db, $query);
		$num_rows = mysqli_num_rows($result);
		for ($i = 0; $i < $num_rows; $i++){
			$row = mysqli_fetch_assoc($result);
			echo '<tr>
				  <td style = "text-align: center">'.$row['doctorname'].'</td>
				  <td style = "text-align: center">'.$row['specialty'].'</td>
				  <td style = "text-align: center">'.$row['available'].'</td>';
			if ($user != 'user'){
				echo '<td style = "text-align: center">
					  <a href = "/db_entry/doctors_profile_form.php?profile='.$row['doctorid'].'">
  					  <img src="/images/view.png" width=45 height=20></a></td>
  					  <td style = "text-align: center">
					  <a href = "/db_entry/doctors_profile_form.php?delete='.$row['doctorid'].'" 
					   onclick="return confirm(\'Are you sure you wish to delete this doctor form your cllinic?\')">
  					  <img src="/images/delete.png" width=15 height=15></a></td>
					  </td>
		 	  	 	  </tr>';
			} else {
				if ($row['available'] != 'N/A'){
					echo '<td style = "text-align: center">'.$row['patients'].'</td>
				   		  <td style = "text-align: center">
						  <a href = "/clinic/wl_signup_form.php?doctorid='.$row['doctorid'].'&
		  				   clinicid='.$this->clinicid.'">
						  <img src="/images/arrange.png" width=60 height=20></a></td>
		 		  	 	  </tr>';
				}
			}
		}
		echo '</table>';
		if ($user != 'user'){
			echo '<br />
				  <table width = "650">
				  <tr><td colspan = 5 style = "text-align: center">
				  <a href = "/db_entry/doctor_form.php">Register New Doctor</a></td></tr>
				  </table>';
		}
	}
	
	function reset_avail_doctors(){
		$query = "select * from doctors where clinicid = ".$this->clinicid;
		$result = mysqli_query($this->db, $query);
		$num_rows = mysqli_num_rows($result);
		for ($i = 0; $i < $num_rows; $i++){
			$row = mysqli_fetch_assoc($result);
			doctor::set_available($this->db, $row['doctorid']);
		}
	}
}

class doctor{
	
	public $db, $doctor_info, $doctorid;
	
	function __construct($doctor_info){
		$this->db = db_connect();
		$this->doctor_info = $doctor_info;
	}
	
	static function get_doctor_info($db, $doctorid){
		$query = "select * from doctors where doctorid = ".$doctorid;
		$result = mysqli_query($db, $query);
		$row = mysqli_fetch_assoc($result);
		return $row;
	}
	
	function set_doctorid(){
		$query = "select doctorid from doctors where doctorname = '".$this->doctor_info['doctorname']."'";
		$result = $this->db->query($query);
		$this->doctorid = $result->fetch_row()[0];
	}
	
	static function lookup_doctor($db, $doctorname){
		$query = "select * from doctors where doctorname = '".$doctorname."'";
		$result = mysqli_query($db, $query);
		$num_rows = mysqli_num_rows($result);
		$row = mysqli_fetch_assoc($result);
	
		if (!$num_rows == 0){
			return $row;
		}
		return false;
	}
	
	static function get_doctorname($db, $wl_id){
		$query = "select doctorname from waitlist where waitlistid=$wl_id";
		$result = $db->query($query);
		return $result->fetch_row()[0];
	}
	
	function register_doctor(){
		$query = "insert into doctors (doctorname, birthdate, gender,
				email, phonenumber, specialty, clinicid) values ('";
		$info_str = implode("', '", $this->doctor_info);
		$query .= $info_str."')";
		$result = mysqli_query($this->db, $query);
		if (!$result){
			throw new Exception("An error has occurred. Doctor could not be added to doctors database.
								 Please try again later.");
		}
		$this->set_doctorid();
		$schedule = array();
		for ($i=0; $i < 7; $i++){
			array_push($schedule, "from 0 to 0");
		}
		$this->set_schedule($this->db, $this->doctorid, $schedule);
		$this->set_available($this->db, $this->doctorid);
		return true;
	}
	
	static function delete_doctor($db, $doctorid){
		$query = "delete from doctors where doctorid=$doctorid";
		$result = $db->query($query);
		if (!$result){
			throw new Exception("Doctor could not be deleted.");
		}
		$query = "delete from doctor_hours where doctorid=$doctorid";
		$result = $db->query($query);
		
		return true;
	}
	
	static function update_doctor($db, $doctorid, $doctor_info){
		$query = "update doctors set ";
		$i = 1;
		foreach ($doctor_info as $key => $value){
			$query .= $key." = '".$value."'";
			if ($i != count($doctor_info)){
				$query .= ", ";
				$i++;				
			} else {
				$query .= " where doctorid = ".$doctorid;
			}
		}
		$result = mysqli_query($db, $query);
		if (!$result){
			throw new Exception("Doctor's new profile could not be updated. Please try again later.");
		}
		return true;
	}
	
	static function update_num_patients($db, $doctor, $plusOrMinusOne){
		if ($doctor != ''){
			$query = "select patients from doctors where doctorname='".$doctor."'";
			$result = $db->query($query);
			$patients = $result->fetch_row()[0];
			$patients += $plusOrMinusOne;
			$query = "update doctors set patients=$patients where doctorname='".$doctor."'";
			$result = $db->query($query);
			if (!$result){
				throw new Exception("Number of patients for the doctor could not be updated.");
			}
		}
	}
	
	static function check_schedule($db, $doctorid){
		$query = "select * from doctor_hours where doctorid = ".$doctorid;
		$result = mysqli_query($db, $query);
		$num_rows = mysqli_num_rows($result);
		if ($num_rows > 0){
			return true;
		}
		return false;
	}
	static function set_schedule($db, $doctorid, $schedule){
		#need doctor_hours table for retrieving specific weekday when checking availability.
		
		$query = "select * from doctor_hours where doctorid = ".$doctorid;
		$result = mysqli_query($db, $query);
		$num_rows = mysqli_num_rows($result);
		if ($num_rows > 0){
			$query = "delete from doctor_hours where doctorid = ".$doctorid;
			$result = mysqli_query($db, $query);
		}
		$query = "insert into doctor_hours values (".$doctorid.", '".implode("', '", $schedule)."')";
		$result = mysqli_query($db, $query);
		if (!$result){
			throw new Exception("Schedule could not be added to doctor_hours database. Please try again later.");
		}
		#if schedule is inserted, retrieve them, turn into a readable schedule string.
		#then insert into doctors table.
		$query = "select * from doctor_hours where doctorid = ".$doctorid;
		$result = mysqli_query($db, $query);
		$row = mysqli_fetch_assoc($result);

		$doctors_schedule = '';
		foreach ($row as $key => $value){
			if ($key != 'doctorid'){
				$doctors_schedule.=$key.': '.$value.'\n';
			}
		}
		
		$query = "update doctors
				  set schedule = '".$doctors_schedule."'
				  where doctorid = ".$doctorid;
		$result = mysqli_query($db, $query);
		if (!$result){
			throw new Exception('Schedule is added to doctor_hours database, 
								 but it could not be set to doctors profile.');
		}
		return true;
	}
	
	static function set_available($db, $doctorid){
		date_default_timezone_set('America/Toronto');
		$datetime = getdate();
		$query = "select * from doctor_hours where doctorid = ".$doctorid;
		$result = mysqli_query($db, $query);
		$row = mysqli_fetch_assoc($result);
		$schedule = array();
		foreach ($row as $key => $value){
			if ($key != 'doctorid'){
				array_push($schedule, $value);
			}
		}
		foreach ($schedule as $key => $value){
			$value = explode(' ', $value);
			$value = range($value[1], $value[3]);
			#if current hour is within the range of doctor's hours today, set to avail now.
			if ($key == $datetime['wday'] && in_array($datetime['hours'], $value)){
				$available = 'Available now';
			}
		}
		if (!isset($available)){
			$available = 'N/A';
		}
		$query = "update doctors 
  				  set available = '".$available."'
  				  where doctorid = ".$doctorid;
		$result = mysqli_query($db, $query);
		if (!$result){
			throw new Exception("Doctor's availablility could not be set. Please try again later.");
		}
		return true;
	}
}

?>