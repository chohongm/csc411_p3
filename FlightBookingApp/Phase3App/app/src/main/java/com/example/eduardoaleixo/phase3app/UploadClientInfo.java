package com.example.eduardoaleixo.phase3app;

import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;
import flightSystem.FileFormatException;
import flightSystem.FlightSystem;
import java.io.FileNotFoundException;

/**
 * An activity that allows admins to upload client information from file.
 */
public class UploadClientInfo extends AppCompatActivity {

    /**
     * Initializes the UploadClientInfo layout.
     * @param savedInstanceState Contains the data it most recently supplied in
     *                           onSaveInstanceState(Bundle)
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_upload_client_info);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

    }

    /**
     * Uploads client information from the filename given by the user when the
     * relevant button is pressed.
     * @param view The view of this activity.
     */
    public void uploadClientInfo(View view) {
        EditText filename =
                (EditText) findViewById(R.id.editText_loadClientInfo);
        TextView status =
                (TextView) findViewById(R.id.textView_clientInfoStatus);

        try {
            //try to open the file
            FlightSystem.getInstance().loadClients(this.getFilesDir().toString()
                    + "/" + filename.getText().toString());

            status.setText(R.string.success_clients_uploaded);
        } catch (FileFormatException e) {
            status.setText(e.getMessage());
        } catch (FileNotFoundException e) {
            status.setText(R.string.file_not_found);
        } catch (Exception e) {
            status.setText(e.getMessage());
        }


    }
}
