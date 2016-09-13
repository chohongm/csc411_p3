<?php
require_once('db_fns.php');

class patient {
	public $patient_info, $query, $result, $num_rows, $row, $patientid;
	
	function __construct($patient_info){
		$this->patient_info = $patient_info;
	}
	
	//search patients database for entered patient info.
	static function lookup_patient($db, $healthcard){
		$query = "select * from patients where healthcardnumber = '".$healthcard."'";
		$result = mysqli_query($db, $query);
		$num_rows = mysqli_num_rows($result);
	
		if ($num_rows > 0){
			$row = mysqli_fetch_assoc($result);
			return $row;
		}
		return false;
	}
	
	function register_patient($db){
		$this->query = "insert into patients values (null, '";
		foreach ($this->patient_info as $key => $value){
			if ($key != 'medicalhistory'){
				$this->query .= $value."', '";
			} else {
				$this->query .= $value."')";
			}
		}
		$this->result = mysqli_query($db, $this->query);
	
		if ($this->result){
			$query = "select * from patients where healthcardnumber = '".$this->patient_info['healthcardnumber']."'";
			$result = mysqli_query($db, $query);
			$this->row = mysqli_fetch_assoc($result);
			$this->patientid = $this->row['patientid'];
			echo "<br /> Patient is successfully added to patients database.<br/>
				  Your patient ID is ".$this->patientid.".<br />";
		} else {
			throw new Exception("Patient could not be registered. Please try again later.");
		}
	}
		
	static function update_patient($db, $patientid, $list){
		$query = "update patients set ";
		$i = 1;
		foreach ($list as $key => $value){
			$query .= $key." = '".$value."'";
			if ($i != count($list)){
				$query .= ", ";
				$i++;				
			} else {
				$query .= " where patientid = ".$patientid;
			}
		}
		return $result = mysqli_query($db, $query);
	}
	
	static function get_patientname($db, $patientid){
		$query = "select patientname from patients where patientid = ".$patientid;
		$result = mysqli_query($db, $query);
		$row = mysqli_fetch_assoc($result);
		return $row['patientname'];
	}
	
	static function get_patient_info($db, $userid){
		$query = "select * from patients as p, authorized_users as u
				  where u.userid = '".$userid."' and u.patientid = p.patientid";
		$result = mysqli_query($db, $query);
		$row = mysqli_fetch_assoc($result);
		return $row;
	}
	
	static function check_if_on_wl($db, $userid){
		$query = "select wl.clinicid from authorized_users as u, waitlist as wl
				  where u.userid = '".$userid."'
				  and u.patientid = wl.patientid";
		$result = mysqli_query($db, $query);
		if (mysqli_num_rows($result) > 0){
			return true;
		}
		return false;
	}
}

class admin extends patient{
	public $ci, $ai;
	
	function __construct($patient_info, $clinicid = NULL){
		
		parent::__construct($patient_info);
		if ($clinicid != null){
			$this->ci = $clinicid;
		}
	}
	//if the account creator is an administrator, by using patient info that was created just now, search for his/her adminid in clinicadmins database.
	function lookup_admin($db){
		$this->query = "select a.adminid, p.patientid from clinicadmins as a, patients as p 
						where p.healthcardnumber ='".$this->patient_info['healthcardnumber']."' 
						and a.patientid = p.patientid and a.adminid != 0";
		$this->result = mysqli_query($db, $this->query);
		$this->num_rows = mysqli_num_rows($this->result);
		//if the admin has yet his/her adminid, add it to clinicadmins database.
		if (!$this->num_rows == 0){
			$row = $this->result->fetch_assoc();
			$this->ai = $row['adminid'];
			return true;
		}
		return false;
	}
	
	static function get_clinicid($db, $userid){
		$query = "select clinicid from clinicadmins, authorized_users
				  where userid = '".$userid."'";
		$result = mysqli_query($db, $query);
		if (mysqli_num_rows($result) != 0){
			$row = mysqli_fetch_assoc($result);
			return $row['clinicid'];
		} else {
			return false;
		}
	}
	
	function register_admin($db){
		if($this->lookup_patient($db, $this->patient_info['healthcardnumber'])){
			$this->row = mysqli_fetch_assoc($this->result);
			$this->query = "insert into clinicadmins values
				(null, ".$this->ci.", ".$this->patientid.")";
			$this->result = mysqli_query($db, $this->query);
			if ($this->result){
				$this->lookup_admin($db);
			} else {
				throw new Exception("Administrator could not be registered. Please try again later.");
			}
		} else {
			throw new Exception("User must be registered as a patient before registering as an admin.");
		}
	}
}

class user extends admin
{
	public $an, $pw;
	
	function __construct($accountname, $password, $patient_info, $clinicid = NULL){
		
		parent::__construct($patient_info, $clinicid);
		$this->an = $accountname;
		$this->pw = $password;
	}
		
	static function get_patientid($db, $userid){
		$query = "select patientid from authorized_users where userid = '".$userid."'";
		$result = mysqli_query($db, $query);
		$row = mysqli_fetch_assoc($result);
		$patientid = $row['patientid'];
		return $patientid;
	}
	
	static function get_postalcode($db, $userid){
		$query = "select p.postalcode from authorized_users as u, patients as p where 
				  u.userid = '".$userid."'
				  and u.patientid = p.patientid";
		$result = mysqli_query($db, $query);
		$row = mysqli_fetch_assoc($result);
		$postalcode = $row['postalcode'];
		return $postalcode;
	}
	
	function register_account($db){

		#register patient if yet registered.
		if(!$user_found = $this->lookup_patient($db, $this->patient_info['healthcardnumber'])){
			$this->register_patient($db);
		} else {
			$this->patientid = $user_found['patientid'];
		}
		
		#if clinicid is entered, user is an admin.
		if (isset($this->ci)){
			#Register admin if yet registered. Query account.
			if(!$this->lookup_admin($db)){
				$this->register_admin($db);
			}
			$query = "insert into authorized_users values
					('".$this->an."', sha1('".$this->pw."'), '".$this->patient_info['email']."',
					  ".$this->patientid.", ".$this->ai.", now())";
		} else {
			#user is not an admin. Query account.
			$query = "insert into authorized_users values
					('".$this->an."', sha1('".$this->pw."'), '".$this->patient_info['email']."',
					  ".$this->patientid.", 'null', now())";
		}
		#attempt to insert new account.
		$result = mysqli_query($db, $query);
		
		#if account created, send email.
		if ($result){
			$query = "select datecreated from authorized_users where userid = '".$this->an."'";
			$result = $db->query($query);
			$date = $result->fetch_row();
			
$to = $this->patient_info['email'];
$from = "From: support@statqueue.com";
$title = 'Your account info';
$message = 'Welcome to StatQueue!
					
Your new account\'s info is as follows,
					
	Date Created: '.reformat_date($date[0]).'
	Accountname: '.$this->an.'
	Patient ID: '.$this->patientid;
if (isset($this->ci)){
$message .= '
	Clinic ID: '.$this->ci.'
	Admin ID: '.$this->ai;
}

			mail($to, $title, $message, $from);
		} else {
			throw new Exception("Your account could not be created. Please try again later.");
		}
		return true;
	}
}

function login($userid, $pw) {
// check username and password with db
// if yes, return true
// else throw exception

  // connect to db
  $db = db_connect();

  // check if username is unique
  $query = "select * from authorized_users where userid='".$userid."'
  		 and password = sha1('".$pw."')";
  $result = mysqli_query($db, $query);
  
  if (!$result) {
     throw new Exception('Could not log you in.');
  }

  if (mysqli_num_rows($result) > 0) {
     return $result->fetch_assoc();
  } else {
     throw new Exception('Either the user ID does not exist or its password is incorrect.');
  }
}

function check_valid_user() {
// see if somebody is logged in and notify them if not
  if (isset($_SESSION['user'])) {
  	return 'user';
  } else if (isset($_SESSION['admin'])) {
  	return 'admin';
  } else if (isset($_SESSION['master'])){
  	return 'master';
  } else {
     // they are not logged in
     return false;
  }
}

function change_pw($userid, $old_pw, $new_pw) {
// change password for username/old_password to new_password
// return true or false

  // if the old password is right
  // change their password to new_password and return true
  // else throw an exception
  login($userid, $old_pw);
  $db = db_connect();
  $result = mysqli_query($db, "update authorized_users
                          set password = sha1('".$new_pw."')
                          where userid = '".$userid."'");
  if (!$result) {
    throw new Exception('Password could not be changed.');
  } else {
    return true;  // changed successfully
  }
}

function get_random_pw($min_len = 6, $max_len = 13) {
// grab a random word from dictionary between the two lengths
// and return it

   // generate a random word
  $word = array_merge(range('a','z'), range('A','Z'), range(0,9));
  shuffle($word);
  return substr(implode($word), 0, rand($min_len, $max_len));
}

function reset_pw($userid) {
// set password for username to a random value
// return the new password or false on failure
  $new_pw = get_random_pw(6, 13);
  
  if($new_pw == false) {
    throw new Exception('Could not generate new password.');
  }

  // set user's password to this in database or return false
  $db = db_connect();
  $result = mysqli_query($db, "update authorized_users
                          set password = sha1('".$new_pw."')
                          where userid = '".$userid."'");
  if (!$result) {
    throw new Exception('Could not change password.');  // not changed
  } else {
    return $new_pw;  // changed successfully
  }
}

function notify_pw($userid, $email, $pw) {
// notify the user that their password has been changed
    $db = db_connect();
    $query = "select email from authorized_users where userid='$userid' and email='$email'";
    $result = mysqli_query($db, $query);
    if (!$result) {
      throw new Exception('Could not connect to database.');
    } else if (mysqli_num_rows($result) == 0) {
      throw new Exception('Could not find email address matching user.');
      // username not in db
    }
      $row = $result->fetch_assoc();
      $email_to = $row['email'];
      $email_from = "From: support@statqueue.com \r\n";
      $title = 'StatQueue: New Password';
      $mesg = "Your new password is ".$pw.".\r\n Please change it to a new password that you will remember the next time you log in.\r\n";

      if (mail($email_to, $title, $mesg, $email_from)) {
        return true;
      } else {
        throw new Exception('Could not send email.');
      }
}

function get_date($city){
	$city = "America/".$city;
	date_default_timezone_set($city);
	// Prints something like: Monday 8th of August 2005 03:12:46 PM
	echo date('l jS \of F Y h:i:s A');
}

?>
