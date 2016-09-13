<?php
$DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
require_once("$DOCUMENT_ROOT/functions/statq_fns.php");
do_header();

if (!filled_out($_POST)){
	throw new Exception("You have not entered all the required details.<br />"
			."Please go back and try again.");
}

if (!valid_account($_POST['accountname'])){
	throw new Exception("Accountname you entered is either invalid or already exists.");
}

if (!valid_pw($_POST['password'])){
	throw new Exception("Your password must consist of at least one in each
						 lowercase letter, uppercase letter, and number.");
}

if ($_POST['password'] != $_POST['password2']){
	throw new Exception("The passwords you have entered do not match.");
}

if (!valid_birthdate($_POST['birthdate'])){
	throw new Exception("Invalid birthdate.");
}

if (!valid_email($_POST['email'])){
	throw new Exception("The email you have entered is not valid.");
}

if (!valid_postalcode($_POST['postalcode'])){
	throw new Exception("Postal code you have entered is not valid.");
}

#set accountname, password, and patient info.
$accountname = $_POST['accountname'];
$password = $_POST['password'];

$patient_info['fullname'] = $_POST['firstname']." ".$_POST['lastname'];

foreach ($_POST as $key => $value){
	if ($key != 'accountname' && $key != 'password' && $key != 'password2' &&
			$key != 'lastname' && $key != 'firstname' && $key != 'clinicid'){
		$patient_info[$key] = $value;
	}
}
if (!get_magic_quotes_gpc()){
	foreach ($patient_info as $value){
		$value = addslashes($value);
	}
}

$db = db_connect();

#if clinicid is entered and clinicid is valid, create user instance as admin.
if ($_POST['clinicid'] != '' && valid_clinicid($_POST['clinicid'] == false)){
	throw new Exception("The clinicid ID you entered is invalid.");
} else if ($_POST['clinicid'] != '' && valid_clinicid($_POST['clinicid'] == true)){
	$clinicid = $_POST['clinicid'];
	$user = new user($accountname, $password, $patient_info, $clinicid);
#if not, create as regular user.
} else {
	$user = new user($accountname, $password, $patient_info);
}

#register user.
$user->register_account($db);

#log-in to new account.
echo "<meta http-equiv=refresh content=0;URL='/user_auth/login.php?userid=$accountname&password=$password'>";

do_footer();
?>